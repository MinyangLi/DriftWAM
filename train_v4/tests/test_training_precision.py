"""Tiny-update, loading, and real checkpoint round-trip precision checks."""
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
from train_v4.engine.precision import assert_fp32_model, assert_fp32_optimizer
from train_v4.engine.resume_state import capture_rng_state, load_resume_state, save_resume_state
from train_v4.engine.trainer import JointDistillationTrainer
from train_v4.model_setup import load_online_student, load_target_student, load_frozen_action_teacher
from train_v4.runtime import activate_lingbot_va

PROJECT = Path(__file__).resolve().parents[2]


def model_and_optimizer():
    model = torch.nn.Linear(1, 1, bias=False)
    model.config = {}
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0, fused=True)
    return model, ema, optimizer


def update(model, ema, optimizer, step):
    optimizer.param_groups[0]['lr'] = 2e-6 * min(1, step / 100)
    model.weight.grad = torch.full_like(model.weight, .25 + step / 1000)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    update_ema(ema, model, decay=.995)


class TrainingPrecisionTest(unittest.TestCase):
    def test_small_adam_updates_and_ema_accumulate_in_fp32(self):
        model, ema, optimizer = model_and_optimizer()
        before = model.weight.detach().clone()
        for step in range(1, 251):
            update(model, ema, optimizer, step)
        assert_fp32_model(model)
        assert_fp32_model(ema)
        assert_fp32_optimizer(optimizer)
        self.assertGreater((before-model.weight).abs().max().item(), 1e-4)
        self.assertGreater((before-ema.weight).abs().max().item(), 1e-5)
        with self.assertRaisesRegex(ValueError, 'FP32'):
            update_ema(ema.bfloat16(), model.bfloat16())

    def test_online_and_ema_load_fp32_but_teacher_keeps_compute_dtype(self):
        activate_lingbot_va(PROJECT/'lingbot-va')
        with tempfile.TemporaryDirectory() as folder:
            config = SimpleNamespace(lingbot_va_root=PROJECT/'lingbot-va',
                                     selected_student_init_path=Path(folder),
                                     teacher_model_path=Path(folder), param_dtype='bfloat16')
            def loader(path, *, torch_dtype, torch_device, attn_mode):
                return torch.nn.Linear(1, 1, dtype=torch_dtype)
            with patch('wan_va.modules.utils.load_transformer', side_effect=loader):
                online = load_online_student(config)
                ema = load_target_student(config)
                teacher = load_frozen_action_teacher(config)
        assert_fp32_model(online)
        assert_fp32_model(ema)
        self.assertFalse(next(ema.parameters()).requires_grad)
        self.assertEqual(next(teacher.parameters()).dtype, torch.bfloat16)

    def test_save_resume_preserves_small_updates_and_next_update(self):
        model, ema, optimizer = model_and_optimizer()
        loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
        for step in range(1, 126):
            update(model, ema, optimizer, step)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            trainer = object.__new__(JointDistillationTrainer)
            trainer.config = SimpleNamespace(rank=0)
            trainer._save_model(model, checkpoint/'online_student')
            trainer._save_model(ema, checkpoint/'target_student')
            save_resume_state(model, optimizer, checkpoint,
                              capture_rng_state(torch.device('cpu'), loader))
            resumed, resumed_ema, resumed_optimizer = model_and_optimizer()
            for target, subdir in ((resumed, 'online_student'), (resumed_ema, 'target_student')):
                state = load_file(checkpoint/subdir/'diffusion_pytorch_model.safetensors')
                self.assertEqual(state['weight'].dtype, torch.float32)
                target.load_state_dict(state)
            load_resume_state(resumed, resumed_optimizer, checkpoint)
            for step in range(126, 151):
                update(model, ema, optimizer, step)
                update(resumed, resumed_ema, resumed_optimizer, step)
        torch.testing.assert_close(resumed.weight, model.weight, rtol=0, atol=0)
        torch.testing.assert_close(resumed_ema.weight, ema.weight, rtol=0, atol=0)
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(resumed_optimizer.state[resumed.weight][key],
                                       optimizer.state[model.weight][key], rtol=0, atol=0)
        assert_fp32_optimizer(resumed_optimizer)


if __name__ == '__main__':
    unittest.main()
