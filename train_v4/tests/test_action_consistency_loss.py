"""Direct CPU coverage for the v4 action-consistency objective."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_v4.forwards.action_signature import ActionSignatureCondition
from train_v4.objectives.action_consistency_loss import ActionConsistencyLoss
from train_v4.objectives.action_training_pair import ActionTrainingPair


class _VelocityModel(torch.nn.Module):
    def __init__(self, scale: float, *, trainable: bool) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(scale),
            requires_grad=trainable,
        )
        self.patch_size = (1, 1, 1)
        self.calls = 0

    def forward(self, input_dict, *, train_mode):
        if not train_mode:
            raise AssertionError("action objective must use train_mode=True")
        self.calls += 1
        action = input_dict["action_dict"]["noisy_latents"]
        batch, channels, frames, positions, _ = action.shape
        velocity = action * self.scale
        action_sequence = (
            velocity.squeeze(-1)
            .permute(0, 2, 3, 1)
            .reshape(batch, frames * positions, channels)
        )
        return torch.zeros(batch, 1, 1), action_sequence


def _mesh_id(
    frames: int,
    height: int,
    width: int,
    token_type: int,
    *,
    f_w: int,
    f_shift: int,
    action: bool,
) -> torch.Tensor:
    del token_type, f_w, f_shift, action
    return torch.zeros(frames * height * width, 3, dtype=torch.long)


class ActionConsistencyLossTest(unittest.TestCase):
    def test_three_forwards_and_online_gradient(self) -> None:
        teacher = _VelocityModel(0.7, trainable=False)
        target = _VelocityModel(0.4, trainable=False)
        online = _VelocityModel(0.1, trainable=True)
        config = SimpleNamespace(
            param_dtype="float32",
            action_num_train_timesteps=1000,
            action_consistency_stride=500,
            action_huber_c=0.001,
            action_dim=30,
            signature_horizon=1,
            frame_chunk_size=2,
            action_per_frame=16,
            attn_window=72,
        )
        objective = object.__new__(ActionConsistencyLoss)
        objective.teacher = teacher
        objective.online_student = online
        objective.target_student = target
        objective.config = config
        objective._get_mesh_id = _mesh_id
        objective._action_channel_ids = tuple(range(16))
        objective.action_scheduler = SimpleNamespace(
            sigmas=torch.linspace(1.0, 0.0, 1000),
            timesteps=torch.linspace(1000.0, 1.0, 1000),
        )

        video = torch.full(
            (1, 2, 2, 1, 1),
            0.25,
            requires_grad=True,
        )
        clean_action = torch.full(
            (1, 30, 2, 16, 1),
            0.5,
            requires_grad=True,
        )
        action_mask = torch.zeros_like(clean_action, dtype=torch.bool)
        action_mask[:, :16] = True
        pair = ActionTrainingPair(
            video=video,
            clean_action=clean_action,
            action_mask=action_mask,
        )
        condition = ActionSignatureCondition(
            text_emb=torch.zeros(1, 1, 1),
            frame_start=torch.tensor([0]),
            history_frame_start=torch.tensor([0]),
        )

        with patch(
            "torch.randint",
            return_value=torch.tensor([100, 400], dtype=torch.long),
        ):
            result = objective(pair, condition)
        total = result.consistency_loss + 0.01 * result.flow_matching_loss
        total.backward()

        self.assertGreater(result.sigma_start[0, 0].item(), 0.)
        self.assertGreater(result.sigma_end[0, 0].item(), 0.)
        self.assertTrue(torch.isfinite(result.consistency_loss))
        self.assertTrue(torch.isfinite(result.flow_matching_loss))
        self.assertEqual(teacher.calls, 1)
        self.assertEqual(target.calls, 1)
        self.assertEqual(online.calls, 1)
        self.assertIsNone(teacher.scale.grad)
        self.assertIsNone(target.scale.grad)
        self.assertIsNone(video.grad)
        self.assertIsNone(clean_action.grad)
        self.assertIsNotNone(online.scale.grad)
        self.assertTrue(torch.isfinite(online.scale.grad))
        self.assertNotEqual(online.scale.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
