"""CPU contracts for teacher-action-probed velocity geometry."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from train_v4.forwards.action_response import ActionResponseProbe
from train_v4.forwards.action_signature import ActionSignatureCondition
from train_v4.geometry.action_response_metric import ActionResponseMetric


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.arange(1000, dtype=torch.float32)

    @staticmethod
    def add_noise(clean, noise, timesteps, *, t_dim):
        if t_dim != 2:
            raise AssertionError("action time must use the frame axis")
        return clean + noise


class _RecordingProbe(ActionResponseProbe):
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            param_dtype="float32",
            signature_horizon=1,
            frame_chunk_size=2,
            action_dim=2,
            action_per_frame=1,
            action_num_train_timesteps=1000,
        )
        self.teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.action_scheduler = _FakeScheduler()
        self.cache_prefix = "test_response"
        self._action_channel_ids = (0, 1)
        self.calls = []

    def _evaluate_velocity(
        self,
        videos,
        noisy_actions,
        action_timesteps,
        text_emb,
        frame_starts,
        history_video,
        history_action,
        history_frame_starts,
        channel_mask,
        *,
        cache_name,
    ):
        self.calls.append(
            {
                "videos": videos.clone(),
                "noisy_actions": noisy_actions.clone(),
                "timesteps": action_timesteps.clone(),
                "cache_name": cache_name,
            }
        )
        video_marker = videos[:, :1, :1, :1, :1].reshape(-1, 1, 1, 1, 1)
        return noisy_actions + video_marker


class ActionResponseTest(unittest.TestCase):
    def test_each_teacher_action_probes_every_student(self) -> None:
        probe = _RecordingProbe()
        teacher_videos = torch.zeros(1, 2, 1, 2, 1, 1)
        teacher_videos[:, 0] = 100
        teacher_videos[:, 1] = 200
        student_videos = torch.zeros(1, 2, 1, 2, 1, 1, requires_grad=True)
        with torch.no_grad():
            student_videos[:, 0] = 300
            student_videos[:, 1] = 400
        teacher_actions = torch.empty(1, 2, 2, 2, 1, 1)
        teacher_actions[:, 0] = 1
        teacher_actions[:, 1] = 5
        shared_noise = torch.full((1, 2, 2, 1, 1), 10.0)
        timestep_ids = torch.tensor([3, 7], dtype=torch.long)
        condition = ActionSignatureCondition(
            text_emb=torch.zeros(1, 1, 1),
            frame_start=torch.tensor([2]),
            history_frame_start=torch.tensor([0]),
        )

        result = probe(
            teacher_videos,
            student_videos,
            teacher_actions,
            shared_noise,
            timestep_ids,
            condition,
        )

        self.assertEqual(result.teacher_responses.shape, (1, 2, 2, 2, 1, 1))
        self.assertEqual(result.student_responses.shape, (1, 2, 2, 2, 2, 1, 1))
        torch.testing.assert_close(
            result.noisy_teacher_actions[:, 0],
            torch.full((1, 2, 2, 1, 1), 11.0),
        )
        torch.testing.assert_close(
            result.noisy_teacher_actions[:, 1],
            torch.full((1, 2, 2, 1, 1), 15.0),
        )
        torch.testing.assert_close(
            result.teacher_responses[:, 0],
            torch.full((1, 2, 2, 1, 1), 111.0),
        )
        torch.testing.assert_close(
            result.teacher_responses[:, 1],
            torch.full((1, 2, 2, 1, 1), 215.0),
        )
        torch.testing.assert_close(
            result.student_responses[:, 0, 0],
            torch.full((1, 2, 2, 1, 1), 311.0),
        )
        torch.testing.assert_close(
            result.student_responses[:, 0, 1],
            torch.full((1, 2, 2, 1, 1), 315.0),
        )
        self.assertEqual(len(probe.calls), 2)
        self.assertEqual(
            probe.calls[0]["videos"][:, 0, 0, 0, 0].tolist(),
            [100.0, 300.0, 400.0],
        )
        self.assertEqual(
            probe.calls[1]["videos"][:, 0, 0, 0, 0].tolist(),
            [200.0, 300.0, 400.0],
        )
        self.assertFalse(result.teacher_responses.requires_grad)
        self.assertFalse(result.student_responses.requires_grad)

    def test_native_selected_checkpoint_matches_batched_response_and_gradient(self):
        from pathlib import Path
        from train_v4.tests.test_native_attention import load_native, tiny_model
        from train_v4.objectives.action_execution_loss import ActionExecutionLoss
        native = load_native('lingbot-va/wan_va/modules/model.py', 'response_checkpoint_native')
        devices = [('cpu', torch.float32)]
        if torch.cuda.is_available():
            devices.append(('cuda', torch.bfloat16))
        for device, dtype in devices:
            with self.subTest(device=device, dtype=dtype):
                torch.manual_seed(17)
                teacher = tiny_model(native, dtype).to(device).requires_grad_(False)
                config = SimpleNamespace(
                    lingbot_va_root=Path(__file__).resolve().parents[2] / 'lingbot-va',
                    signature_horizon=1, param_dtype=str(dtype).split('.')[-1],
                    frame_chunk_size=2, action_dim=4, action_per_frame=2,
                    action_num_train_timesteps=10, action_snr_shift=1.0,
                )
                probe = ActionResponseProbe(teacher, config, action_channel_ids=(0, 1, 2, 3))
                def rand(*shape):
                    return torch.randn(*shape, device=device, dtype=dtype)
                teacher_videos = rand(2, 2, 4, 2, 1, 1)
                videos = rand(2, 3, 4, 2, 1, 1).requires_grad_()
                actions, noise = rand(2, 2, 4, 2, 2, 1), rand(2, 4, 2, 2, 1)
                condition = ActionSignatureCondition(
                    text_emb=rand(2, 5, 8), frame_start=4,
                    history_video=rand(2, 4, 4, 1, 1),
                    history_action=rand(2, 4, 4, 2, 1), history_frame_start=0,
                )
                selected = torch.tensor([[0, 1, 0], [1, 0, 1]], device=device)
                full = probe(teacher_videos, videos, actions, noise,
                             torch.tensor([1, 7]), condition, track_student_grad=True)
                expected = ActionExecutionLoss.select_student_responses(full.student_responses, selected)
                expected.float().square().mean().backward()
                from unittest.mock import patch
                from torch.utils.checkpoint import checkpoint
                for budget in (0, 9792):
                    actual_videos = videos.detach().clone().requires_grad_()
                    with patch("train_v4.forwards.action_response._SELECTED_DIRECT_CONTEXT_TOKEN_BUDGET", budget), patch(
                        "train_v4.forwards.action_response.checkpoint", wraps=checkpoint
                    ) as recompute:
                        actual = probe.compute_selected_student_responses(
                            actual_videos, full.noisy_teacher_actions, full.action_timestep_ids,
                            selected, condition,
                        )
                        self.assertEqual(recompute.call_count, 6 if budget == 0 else 0)
                    # Both paths clear caches before backward; fallback rebuilds them.
                    self.assertTrue(all(v is None for b in teacher.blocks for v in b.attn1.attn_caches.values()))
                    actual.float().square().mean().backward()
                    tolerance = dict(rtol=1e-4, atol=1e-6) if dtype == torch.float32 else dict(rtol=0.03, atol=0.003)
                    torch.testing.assert_close(actual, expected, **tolerance)
                    grad_tolerance = dict(rtol=2e-4, atol=1e-7) if dtype == torch.float32 else dict(rtol=0.05, atol=1e-5)
                    torch.testing.assert_close(actual_videos.grad, videos.grad, **grad_tolerance)
                    self.assertGreater(actual_videos.grad.float().norm().item(), 0)
                    self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                    self.assertTrue(all(v is None for b in teacher.blocks for v in b.attn1.attn_caches.values()))

    def test_negative_metric_uses_compared_candidates_matched_probe(self) -> None:
        teacher = torch.tensor([[0.0, 10.0], [0.0, 20.0]]).view(
            2, 2, 1, 1, 1, 1
        )
        student = torch.tensor([
            [[1.0, 10.0], [4.0, 20.0], [9.0, 30.0]],
            [[2.0, 20.0], [8.0, 40.0], [18.0, 60.0]],
        ]).view(2, 3, 2, 1, 1, 1, 1).requires_grad_()
        selected = torch.tensor([[1, 0, 1], [0, 1, 0]])
        metric = ActionResponseMetric(action_dim=1, action_channel_ids=(0,))

        result = metric(teacher, student, selected_teacher=selected)

        torch.testing.assert_close(
            result.student_teacher,
            torch.tensor([
                [[1.0, 0.0], [16.0, 100.0], [81.0, 400.0]],
                [[4.0, 0.0], [64.0, 400.0], [324.0, 1600.0]],
            ]),
        )
        # Query-probe matching would transpose these matrices. Averaging
        # probes or comparing each endpoint's own response also differs.
        torch.testing.assert_close(
            result.student_student,
            torch.tensor([
                [[0.0, 9.0, 400.0], [100.0, 0.0, 100.0], [400.0, 25.0, 0.0]],
                [[0.0, 400.0, 256.0], [36.0, 0.0, 100.0], [256.0, 400.0, 0.0]],
            ]),
        )
        self.assertFalse(result.student_teacher.requires_grad)
        self.assertFalse(result.student_student.requires_grad)
        order = torch.tensor([2, 0, 1])
        reordered = metric(
            teacher, student[:, order], selected_teacher=selected[:, order]
        )
        torch.testing.assert_close(
            reordered.student_student, result.student_student[:, order][:, :, order]
        )

    def test_metric_ignores_inactive_channels_and_invalid_positions(self) -> None:
        teacher = torch.zeros(1, 2, 3, 1, 2, 1)
        student = torch.zeros(1, 2, 2, 3, 1, 2, 1)
        student[:, 0, :, 0, 0, 0] = 2
        student[:, 1, :, 0, 0, 0] = 4
        metric = ActionResponseMetric(action_dim=3, action_channel_ids=(0,))
        valid_mask = torch.tensor([[[True, False]]])
        selected = torch.tensor([[0, 1]])
        expected = metric(teacher, student, valid_mask, selected_teacher=selected)

        teacher[..., 1:, :, :, :] = 1_000_000
        student[..., 1:, :, :, :] = -1_000_000
        teacher[..., 0, 0, 1, 0] = 2_000_000
        student[..., 0, 0, 1, 0] = -2_000_000
        actual = metric(teacher, student, valid_mask, selected_teacher=selected)

        torch.testing.assert_close(actual.student_teacher, expected.student_teacher)
        torch.testing.assert_close(actual.student_student, expected.student_student)
        self.assertTrue(
            torch.equal(
                torch.diagonal(actual.student_student, dim1=1, dim2=2),
                torch.zeros(1, 2),
            )
        )


if __name__ == "__main__":
    unittest.main()
