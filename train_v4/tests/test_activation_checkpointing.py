"""Packed-only recomputation and compatibility with cached training/EMA."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

from train_v4.engine.ema import update_ema
from train_v4.engine.resume_state import capture_rng_state, load_resume_state, save_resume_state
from train_v4.engine.trainer import JointDistillationTrainer
from train_v4.forwards import activation_checkpointing as ac
from train_v4.tests.test_native_attention import load_native, tiny_model, packet


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.calls = 0

    def forward(self, x, *, attention_mask=None):
        self.calls += 1
        return self.linear(x).sin()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.config = {}

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, attention_mask=mask)
        return x


class ActivationCheckpointingTest(unittest.TestCase):
    def test_only_gradient_enabled_packed_calls_recompute(self):
        model = ac.apply_packed_activation_checkpointing(_Model())
        mask = torch.ones(1, 1, dtype=torch.bool)
        model(torch.ones(2, 4), mask).sum().backward()
        self.assertEqual([b.calls for b in model.blocks], [2, 2])
        with torch.no_grad():
            model(torch.ones(2, 4), mask)
        self.assertEqual([b.calls for b in model.blocks], [3, 3])
        model(torch.ones(2, 4)).sum().backward()
        self.assertEqual([b.calls for b in model.blocks], [4, 4])

    def test_native_cached_forward_and_backward_bypass_checkpointing(self):
        module = load_native('lingbot-va/wan_va/modules/model.py', 'native_cached_ac_test')
        torch.manual_seed(31)
        reference = tiny_model(module)
        wrapped = ac.apply_packed_activation_checkpointing(copy.deepcopy(reference))
        inputs = packet(batch=1)
        outputs = []
        for model in (reference, wrapped):
            model.create_empty_cache('test', 72, 2, 4, 'cpu', torch.float32, 1)
            def run(key, start, action, update):
                source = inputs[key]
                positions = 2 if action else 1
                sample = dict(
                    noisy_latents=source['noisy_latents' if update == 0 else 'latent'][:, :, start:start+2],
                    timesteps=source['timesteps' if update == 0 else 'cond_timesteps'][:, start:start+2],
                    grid_id=source['grid_id'][:, :, start*positions:(start+2)*positions],
                    text_emb=inputs['latent_dict']['text_emb'],
                )
                return model(sample, cache_name='test', action_mode=action, update_cache=update)
            with patch.object(ac, 'checkpoint', side_effect=AssertionError('cached calls must not checkpoint')):
                with torch.no_grad():
                    run('latent_dict', 0, False, 2)
                    run('action_dict', 0, True, 2)
                output = run('latent_dict', 2, False, 0)
                model.clear_cache('test')
                output.square().mean().backward()
                outputs.append(output.detach())
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for expected, actual in zip(reference.parameters(), wrapped.parameters()):
            self.assertEqual(expected.grad is None, actual.grad is None)
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_wrapped_ema_and_real_checkpoint_resume(self):
        torch.manual_seed(11)
        model = ac.apply_packed_activation_checkpointing(_Model())
        ema = copy.deepcopy(model).requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
        mask = torch.ones(1, 1, dtype=torch.bool)
        def step(online, target, optim):
            online(torch.ones(2, 4), mask).square().mean().backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            update_ema(target, online)
        step(model, ema, optimizer)
        self.assertEqual(dict(model.named_parameters()).keys(), dict(ema.named_parameters()).keys())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = object.__new__(JointDistillationTrainer)
            trainer.config = SimpleNamespace(rank=0)
            trainer._save_model(model, root/'online_student')
            trainer._save_model(ema, root/'target_student')
            loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
            save_resume_state(model, optimizer, root, capture_rng_state(torch.device('cpu'), loader))
            restored, restored_ema = _Model(), _Model()
            for fresh, name in ((restored, 'online_student'), (restored_ema, 'target_student')):
                state = load_file(root/name/'diffusion_pytorch_model.safetensors')
                self.assertFalse(any('_checkpoint_wrapped_module' in k for k in state))
                fresh.load_state_dict(state)
                ac.apply_packed_activation_checkpointing(fresh)
            restored_ema.requires_grad_(False)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-6)
            load_resume_state(restored, restored_optimizer, root)
            step(model, ema, optimizer)
            step(restored, restored_ema, restored_optimizer)
            for expected, actual in ((model, restored), (ema, restored_ema)):
                for p, q in zip(expected.parameters(), actual.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
