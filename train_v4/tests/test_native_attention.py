"""Causality checks through both real native transformer implementations."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from train_v4.runtime import activate_lingbot_va
from train_v4.forwards.activation_checkpointing import apply_packed_activation_checkpointing

PROJECT = Path(__file__).resolve().parents[2]


def load_native(relative, name):
    activate_lingbot_va(PROJECT / 'lingbot-va')
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tiny_model(module, dtype=torch.float32):
    return module.WanTransformer3DModel(
        patch_size=[1, 1, 1], num_attention_heads=1, attention_head_dim=24,
        in_channels=4, out_channels=4, action_dim=4, text_dim=8, freq_dim=16,
        ffn_dim=32, num_layers=2, attn_mode='torch',
    ).to(dtype).eval()


def packet(batch=2, dtype=torch.float32):
    video = dict(noisy_latents=torch.randn(batch, 4, 4, 1, 1, dtype=dtype),
                 latent=torch.randn(batch, 4, 4, 1, 1, dtype=dtype),
                 timesteps=torch.full((batch, 4), 500.),
                 cond_timesteps=torch.zeros(batch, 4),
                 grid_id=torch.zeros(batch, 3, 4),
                 text_emb=torch.randn(batch, 5, 8, dtype=dtype))
    action = dict(noisy_latents=torch.randn(batch, 4, 4, 2, 1, dtype=dtype),
                  latent=torch.randn(batch, 4, 4, 2, 1, dtype=dtype),
                  timesteps=torch.full((batch, 4), 500.),
                  cond_timesteps=torch.zeros(batch, 4),
                  grid_id=torch.zeros(batch, 3, 8))
    return dict(latent_dict=video, action_dict=action, chunk_size=2, window_size=72)


class NativeAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.modules = [load_native(relative, f'test_native_wan_{i}') for i, relative in enumerate((
            'lingbot-va/wan_va/modules/model.py', 'Flash-WAM/wan_va/modules/model.py'))]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_forbidden_labels_future_and_other_batch_are_invisible(self):
        for module in self.modules:
            cases = [("cpu", torch.float32), ("cpu", torch.bfloat16)]
            if torch.cuda.is_available():
                cases.append(("cuda", torch.bfloat16))
            for device, dtype in cases:
                with self.subTest(module=module.__name__, dtype=dtype, device=device), torch.no_grad():
                    torch.manual_seed(7)
                    model, inputs = tiny_model(module, dtype).to(device), packet(dtype=dtype)
                    for key in ("latent_dict", "action_dict"):
                        inputs[key] = {name: value.to(device) for name, value in inputs[key].items()}
                    reference = model(inputs, train_mode=True)
                    changed = copy.deepcopy(inputs)
                    changed['action_dict']['latent'][:, :, :2] += 5
                    result = model(changed, train_mode=True)
                    torch.testing.assert_close(result[1][:, :4], reference[1][:, :4], rtol=0, atol=0)
                    torch.testing.assert_close(result[0][:, :2], reference[0][:, :2], rtol=0, atol=0)
                    changed = copy.deepcopy(inputs)
                    changed['latent_dict']['latent'][:, :, :2] += 5
                    result = model(changed, train_mode=True)
                    torch.testing.assert_close(result[0][:, :2], reference[0][:, :2], rtol=0, atol=0)
                    # Positive control: current clean video is legal for actions.
                    self.assertGreater((result[1][:, :4] - reference[1][:, :4]).abs().max().item(), 0)
                    for key in ('latent_dict', 'action_dict'):
                        changed = copy.deepcopy(inputs)
                        for field in ('latent', 'noisy_latents'):
                            changed[key][field][:, :, 2:] += 9
                        result = model(changed, train_mode=True)
                        torch.testing.assert_close(result[0][:, :2], reference[0][:, :2], rtol=0, atol=0)
                        torch.testing.assert_close(result[1][:, :4], reference[1][:, :4], rtol=0, atol=0)
                    changed = copy.deepcopy(inputs)
                    for key in ('latent_dict', 'action_dict'):
                        for field in ('latent', 'noisy_latents'):
                            changed[key][field][1] += 7
                    changed['latent_dict']['text_emb'][1] += 11
                    result = model(changed, train_mode=True)
                    for actual, expected in zip(result, reference):
                        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)

    def test_flash_forward_dynamics_keeps_explicit_action_conditioning(self):
        module = self.modules[1]
        torch.manual_seed(29)
        model, inputs = tiny_model(module), packet()
        changed = copy.deepcopy(inputs)
        changed['action_dict']['latent'][:, :, :2] += 5
        with torch.no_grad():
            before = model(inputs, train_mode=True, fdm=True)[0][:, :2]
            after = model(changed, train_mode=True, fdm=True)[0][:, :2]
        self.assertGreater((before-after).abs().max().item(), 1e-5)

    def test_unmasked_negative_control_detects_the_original_leak(self):
        for module in self.modules:
            torch.manual_seed(7)
            model, inputs = tiny_model(module), packet()
            changed = copy.deepcopy(inputs)
            changed['action_dict']['latent'][:, :, :2] += 5
            with patch.object(module.FlexAttnFunc, 'masked_attention',
                              side_effect=lambda q, k, v, mask: module.custom_sdpa(q, k, v)), torch.no_grad():
                before = model(inputs, train_mode=True)[1][:, :4]
                after = model(changed, train_mode=True)[1][:, :4]
            self.assertGreater((before - after).abs().max().item(), 1e-5)

    def test_padding_cannot_influence_valid_attention(self):
        for module in self.modules:
            mask, _ = module.FlexAttnFunc.init_mask(
                (2, 4, 4, 1, 1), (2, 4, 4, 2, 1), 80, 2, 72, (1, 1, 1),
                device='cpu', text_length=5,
            )
            q, k, v = [torch.randn(1, 128, 1, 8) for _ in range(3)]
            reference = module.FlexAttnFunc.masked_attention(q, k, v, mask)
            k[:, 48:] += 1000
            v[:, 48:] += 1000
            actual = module.FlexAttnFunc.masked_attention(q, k, v, mask)
            torch.testing.assert_close(actual[:, :48], reference[:, :48], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())

    def test_checkpoint_recomputation_keeps_its_own_masks(self):
        for module in self.modules:
            torch.manual_seed(19)
            reference = tiny_model(module)
            checkpointed = copy.deepcopy(reference)
            if module is self.modules[0]:
                apply_packed_activation_checkpointing(checkpointed)
            else:
                for i, block in enumerate(checkpointed.blocks):
                    checkpointed.blocks[i] = checkpoint_wrapper(block, preserve_rng_state=False)
            inputs = packet()
            loss = sum(x.square().mean() for x in reference(inputs, train_mode=True))
            loss.backward()
            actual_loss = sum(x.square().mean() for x in checkpointed(inputs, train_mode=True))
            # Another forward changes the packed batch layout before backward.
            with torch.no_grad():
                checkpointed(packet(batch=1), train_mode=True)
            actual_loss.backward()
            torch.testing.assert_close(actual_loss, loss, rtol=0, atol=0)
            for expected, actual in zip(reference.parameters(), checkpointed.parameters()):
                if expected.grad is not None:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)

    def test_packed_training_matches_legal_cached_conditioning(self):
        for module in self.modules:
            torch.manual_seed(23)
            model, inputs = tiny_model(module), packet(batch=1)
            with torch.no_grad():
                packed_video, packed_action = model(inputs, train_mode=True)
                model.create_empty_cache('test', 72, 2, 4, 'cpu', torch.float32, 1)
                def run(key, clean, start, action, update):
                    source = inputs[key]
                    positions = 2 if action else 1
                    sample = dict(noisy_latents=source['latent' if clean else 'noisy_latents'][:, :, start:start+2],
                                  timesteps=source['cond_timesteps' if clean else 'timesteps'][:, start:start+2],
                                  grid_id=source['grid_id'][:, :, start*positions:(start+2)*positions],
                                  text_emb=inputs['latent_dict']['text_emb'])
                    return model(sample, cache_name='test', action_mode=action, update_cache=update)
                run('latent_dict', True, 0, False, 2)
                run('action_dict', True, 0, True, 2)
                video = run('latent_dict', False, 2, False, 0)
                run('latent_dict', True, 2, False, 1)
                action = run('action_dict', False, 2, True, 0)
                model.clear_cache('test')
            torch.testing.assert_close(video, packed_video[:, 2:4], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(action, packed_action[:, 4:8], rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for compiled FlexAttention')
    def test_cuda_flex_matches_dense_forward_and_gradient(self):
        for module in self.modules:
            cpu_masks = module.FlexAttnFunc.init_mask(
                (1, 4, 4, 1, 1), (1, 4, 4, 2, 1), 104, 2, 72, (1, 1, 1),
                device='cpu', text_length=5,
            )
            gpu_masks = module.FlexAttnFunc.init_mask(
                (1, 4, 4, 1, 1), (1, 4, 4, 2, 1), 104, 2, 72, (1, 1, 1),
                device='cuda', text_length=5,
            )
            for mask, dense in zip(gpu_masks, cpu_masks):
                # This layout has one partial block and no full blocks. Keep
                # these checks before backward to catch corrupt sparse indices.
                torch.testing.assert_close(mask.kv_num_blocks, torch.ones_like(mask.kv_num_blocks))
                torch.testing.assert_close(mask.q_num_blocks, torch.ones_like(mask.q_num_blocks))
                torch.testing.assert_close(mask.full_kv_num_blocks, torch.zeros_like(mask.full_kv_num_blocks))
                torch.testing.assert_close(mask.full_q_num_blocks, torch.zeros_like(mask.full_q_num_blocks))
                q = torch.randn(1, 128, 1, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(1, dense.shape[1], 1, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                result = module.FlexAttnFunc.masked_attention(q, k, v, mask)
                reference = module.custom_sdpa(q, k, v, attn_mask=dense.cuda())
                torch.testing.assert_close(result, reference, rtol=.03, atol=.03)
                actual_grads = torch.autograd.grad(result.float().square().mean(), (q, k, v))
                expected_grads = torch.autograd.grad(reference.float().square().mean(), (q, k, v))
                for actual, expected in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, expected, rtol=.05, atol=.001)


if __name__ == '__main__':
    unittest.main()
