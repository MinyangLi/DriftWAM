"""CPU tests for the stage-two action-response training data flow."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_v4.engine.training_data import TrainingBatch
from train_v4.engine.training_step import TrainingStep
from train_v4.engine.trainer import calibrate_bandwidths
from train_v4.engine.teacher_signature_cache import TeacherSignatureBatch
from train_v4.forwards.action_response import ActionResponseBatch, ActionResponseProbe
from train_v4.forwards.action_signature import ActionSignatureCondition
from train_v4.geometry.action_response_metric import (
    ActionResponseMetric,
    ActionResponseMetricResult,
)
from train_v4.geometry.bandwidth import BandwidthCalibrator, KernelBandwidths
from train_v4.geometry.drifting_kernel import DriftingKernel
from train_v4.geometry.kernel_diagnostics import KernelDiagnostics
from train_v4.geometry.video_metric import (
    VideoDistances,
    VideoMetric,
    VideoMetricResult,
)
from train_v4.objectives.action_execution_loss import ActionExecutionLoss
from train_v4.objectives.video_objective import VideoObjective


class _Student(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))


class _StudentVideoGenerator:
    def __init__(self) -> None:
        self.student = _Student()
        self.calls = 0

    def __call__(self, noise, condition, *, initial_frame):
        del condition, initial_frame
        self.calls += 1
        return noise * self.student.scale


class _ActionResponseProbe:
    def __init__(self) -> None:
        self.teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.calls = []
        self.selected_calls = []

    def __call__(
        self,
        teacher_videos,
        student_videos,
        teacher_actions,
        action_noise,
        timestep_ids,
        condition,
        *,
        track_student_grad=False,
    ):
        del condition
        self.calls.append(
            {
                "teacher_shape": tuple(teacher_videos.shape),
                "student_shape": tuple(student_videos.shape),
                "teacher_actions": teacher_actions.detach().clone(),
                "noise": action_noise.detach().clone(),
                "timestep_ids": timestep_ids.detach().clone(),
                "track_student_grad": track_student_grad,
            }
        )
        batch, teacher_count = teacher_videos.shape[:2]
        student_count = student_videos.shape[1]
        _, _, channels, frames, positions, trailing = teacher_actions.shape
        teacher_marker = teacher_videos.float().mean(dim=(2, 4, 5))
        teacher_responses = teacher_marker[:, :, None, :, None, None].expand(
            batch,
            teacher_count,
            channels,
            frames,
            positions,
            trailing,
        )
        student_input = (
            student_videos if track_student_grad else student_videos.detach()
        )
        student_marker = student_input.float().mean(dim=(2, 4, 5))
        student_responses = student_marker[:, :, None, None, :, None, None].expand(
            batch,
            student_count,
            teacher_count,
            channels,
            frames,
            positions,
            trailing,
        )
        probe_offsets = torch.arange(
            teacher_count,
            device=student_videos.device,
            dtype=student_responses.dtype,
        ).view(1, 1, teacher_count, 1, 1, 1, 1)
        student_responses = student_responses + probe_offsets
        return ActionResponseBatch(
            teacher_responses=teacher_responses,
            student_responses=student_responses,
            noisy_teacher_actions=teacher_actions.detach(),
            action_noise=action_noise.detach(),
            action_timestep_ids=timestep_ids.detach(),
        )

    def compute_selected_student_responses(
        self,
        student_videos,
        noisy_teacher_actions,
        timestep_ids,
        selected_teacher,
        condition,
    ):
        del condition
        self.selected_calls.append(
            {
                "student_shape": tuple(student_videos.shape),
                "noisy_actions": noisy_teacher_actions.detach().clone(),
                "timestep_ids": timestep_ids.detach().clone(),
                "selected_teacher": selected_teacher.detach().clone(),
            }
        )
        batch, student_count = student_videos.shape[:2]
        _, _, channels, frames, positions, trailing = (
            noisy_teacher_actions.shape
        )
        student_marker = student_videos.float().mean(dim=(2, 4, 5))
        responses = student_marker[:, :, None, :, None, None].expand(
            batch,
            student_count,
            channels,
            frames,
            positions,
            trailing,
        )
        return responses + selected_teacher.to(
            device=responses.device,
            dtype=responses.dtype,
        )[:, :, None, None, None, None]


class _FakeAttention:
    def init_kv_cache(self, *args, **kwargs):
        del args, kwargs


class _DifferentiableActionTeacher(torch.nn.Module):
    """Minimal frozen teacher preserving video-to-action input gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(
            torch.tensor(0.0),
            requires_grad=False,
        )
        self.patch_size = (1, 1, 1)
        self.num_attention_heads = 1
        self.attention_head_dim = 1
        self.blocks = [SimpleNamespace(attn1=_FakeAttention())]
        self.video_context = {}

    def clear_cache(self, cache_name):
        self.video_context.pop(cache_name, None)

    def forward(self, inputs, *, update_cache, cache_name, action_mode):
        latents = inputs["noisy_latents"]
        if not action_mode:
            self.video_context[cache_name] = latents.mean(dim=(1, 3, 4))
            return torch.zeros_like(latents)
        del update_cache
        action = latents.permute(0, 2, 3, 1, 4).squeeze(-1)
        context = self.video_context[cache_name]
        response = action + context[:, :, None, None]
        return response.reshape(response.shape[0], -1, response.shape[-1])


class _VideoFeatureExtractor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, videos, noise, condition):
        del noise, condition
        self.calls.append(tuple(videos.shape))
        # [B,Q,C,F,H,W] -> [B,Q,F,N=1,D=1]
        return videos.mean(dim=(2, 4, 5)).unsqueeze(-1).unsqueeze(-1)


class _TeacherSignatureCache:
    def __init__(self) -> None:
        self.calls = []

    def get_or_compute(self, sample_ids, videos, condition):
        del condition
        self.calls.append((sample_ids, tuple(videos.shape)))
        batch, candidates, _, frames, _, _ = videos.shape
        signatures = torch.zeros(
            batch,
            candidates,
            1,
            frames,
            1,
            1,
            device=videos.device,
            dtype=videos.dtype,
        )
        return TeacherSignatureBatch(
            signatures=signatures,
            hits=batch,
            misses=0,
        )


class StageTwoDataFlowTest(unittest.TestCase):
    def _config(self, execution_response_mode="off"):
        return SimpleNamespace(
            signature_horizon=1,
            param_dtype="float32",
            student_candidate_count=2,
            teacher_candidate_count=3,
            frame_chunk_size=2,
            action_dim=1,
            action_per_frame=1,
            action_num_train_timesteps=10,
            used_action_channel_ids=(0,),
            video_drifting_loss_weight=1.0,
            beta_rep=1.0,
            execution_response_mode=execution_response_mode,
            execution_loss_weight=1.0,
        )

    def _training_step(self, config, response_probe, student_generator):
        return TrainingStep(
            config,
            KernelBandwidths(video=1.0, action=1.0),
            student_generator,
            response_probe,
            _VideoFeatureExtractor(),
            VideoMetric(),
            ActionResponseMetric(action_dim=1, action_channel_ids=(0,)),
            VideoObjective(config),
            _TeacherSignatureCache(),
            ActionExecutionLoss(action_dim=1, action_channel_ids=(0,)),
            object(),
        )

    def _batch(self):
        teacher_videos = torch.tensor([1.0, 3.0, 5.0]).view(
            1, 3, 1, 1, 1, 1
        ).expand(1, 3, 1, 2, 1, 1).clone()
        return TrainingBatch(
            sample_ids=("sample",),
            teacher_videos=teacher_videos,
            gt_video=torch.full((1, 1, 2, 1, 1), 11.0, requires_grad=True),
            gt_action=torch.full((1, 1, 2, 1, 1), 0.75, requires_grad=True),
            gt_action_mask=torch.ones(1, 1, 2, 1, 1, dtype=torch.bool),
            initial_frame=torch.zeros(1, 1, 1, 1, 1),
            text_emb=torch.zeros(1, 1, 1),
            frame_start=torch.tensor([2]),
            history_video=None,
            history_action=None,
            history_frame_start=torch.tensor([0]),
            video_valid_frames=torch.ones(1, 2, dtype=torch.bool),
            action_valid_mask=torch.ones(1, 1, 2, 1, 1, dtype=torch.bool),
        )

    def test_training_step_generates_students_once_and_reads_teacher_cache(self):
        torch.manual_seed(7)
        config = self._config()
        student_generator = _StudentVideoGenerator()
        response_probe = _ActionResponseProbe()
        feature_extractor = _VideoFeatureExtractor()
        signature_cache = _TeacherSignatureCache()
        step = TrainingStep(
            config,
            KernelBandwidths(video=1.0, action=1.0),
            student_generator,
            response_probe,
            feature_extractor,
            VideoMetric(),
            ActionResponseMetric(action_dim=1, action_channel_ids=(0,)),
            VideoObjective(config),
            signature_cache,
            None,
            object(),
        )

        with patch.object(
            step.action_response_metric,
            "compute_student_student",
            wraps=step.action_response_metric.compute_student_student,
        ) as negative_metric:
            result = step(self._batch(), collect_diagnostics=True)
        # Probe matches are required even when execution is switched off.
        torch.testing.assert_close(
            negative_metric.call_args.args[1],
            result.video_objective.kernel.positive_weights.argmax(dim=-1),
        )
        pair = result.action_consistency_plan.pair
        torch.testing.assert_close(pair.video, self._batch().gt_video)
        torch.testing.assert_close(pair.clean_action, self._batch().gt_action)
        self.assertFalse(pair.video.requires_grad)
        self.assertFalse(pair.clean_action.requires_grad)
        self.assertTrue(pair.action_mask.all())
        self.assertEqual(
            tuple(pair.clean_action.shape), (1, 1, 2, 1, 1)
        )
        assert result.combined_loss is not None
        result.combined_loss.backward()

        self.assertEqual(student_generator.calls, 1)
        self.assertEqual(len(signature_cache.calls), 1)
        self.assertEqual(signature_cache.calls[0][1][1], 3)
        self.assertEqual(len(response_probe.calls), 1)
        self.assertFalse(response_probe.calls[0]["track_student_grad"])
        self.assertEqual(response_probe.calls[0]["teacher_shape"][1], 3)
        self.assertEqual(response_probe.calls[0]["student_shape"][1], 2)
        self.assertGreater(
            torch.count_nonzero(response_probe.calls[0]["noise"]).item(),
            0,
        )
        self.assertEqual(len(feature_extractor.calls), 2)
        self.assertIsNotNone(student_generator.student.scale.grad)
        self.assertTrue(torch.isfinite(student_generator.student.scale.grad))
        self.assertNotIn("loss/video_execution", result.scalars)

    def test_gt_pair_is_independent_of_video_and_execution_inputs(self):
        outputs = []
        for gt_offset, teacher_offset in ((0.0, 0.0), (10.0, 0.0), (0.0, 7.0)):
            torch.manual_seed(37)
            generator = _StudentVideoGenerator()
            probe = _ActionResponseProbe()
            step = self._training_step(self._config("recompute_selected"), probe, generator)
            batch = self._batch()
            batch.gt_video = (batch.gt_video.detach() + gt_offset).requires_grad_(True)
            batch.gt_action = (batch.gt_action.detach() + gt_offset).requires_grad_(True)
            batch.teacher_videos += teacher_offset
            result = step(batch)
            pair = result.action_consistency_plan.pair
            torch.testing.assert_close(pair.video, batch.gt_video)
            torch.testing.assert_close(pair.clean_action, batch.gt_action)
            self.assertFalse(pair.video.requires_grad)
            self.assertFalse(pair.clean_action.requires_grad)
            execution = step.compute_deferred_execution(result.deferred_execution)
            (result.video_loss + execution.loss).backward()
            self.assertIsNone(batch.gt_video.grad)
            self.assertIsNone(batch.gt_action.grad)
            # The cached teacher probes remain separate from the GT actions.
            self.assertTrue(probe.calls[0]["teacher_actions"].eq(0).all())
            outputs.append((result.video_loss.detach(), execution.loss.detach(),
                            generator.student.scale.grad.detach(), pair.video, pair.clean_action))
        # Changing only current GT does not affect either video objective.
        for before, after in zip(outputs[0][:3], outputs[1][:3]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        # Changing teacher candidates does not select a different action target.
        for before, after in zip(outputs[0][3:], outputs[2][3:]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_execution_modes_match_loss_and_student_gradient(self):
        results = {}
        for mode in ("reuse_all", "recompute_selected"):
            torch.manual_seed(7)
            config = self._config(mode)
            student_generator = _StudentVideoGenerator()
            response_probe = _ActionResponseProbe()
            step = self._training_step(
                config,
                response_probe,
                student_generator,
            )

            result = step(self._batch(), collect_diagnostics=True)
            if mode == "reuse_all":
                self.assertIsNotNone(result.execution)
                execution = result.execution
                self.assertIsNotNone(result.combined_loss)
                total_loss = result.combined_loss
            else:
                self.assertIsNone(result.execution)
                self.assertIsNotNone(result.deferred_execution)
                execution = step.compute_deferred_execution(
                    result.deferred_execution
                )
                total_loss = result.video_loss + execution.loss
            torch.testing.assert_close(
                execution.selected_teacher,
                result.video_objective.kernel.positive_weights.argmax(dim=-1),
            )
            total_loss.backward()
            results[mode] = {
                "loss": execution.loss.detach().clone(),
                "total": total_loss.detach().clone(),
                "grad": student_generator.student.scale.grad.detach().clone(),
                "full_calls": response_probe.calls,
                "selected_calls": response_probe.selected_calls,
            }

        torch.testing.assert_close(
            results["reuse_all"]["loss"],
            results["recompute_selected"]["loss"],
        )
        torch.testing.assert_close(
            results["reuse_all"]["total"],
            results["recompute_selected"]["total"],
        )
        torch.testing.assert_close(
            results["reuse_all"]["grad"],
            results["recompute_selected"]["grad"],
        )
        self.assertTrue(
            results["reuse_all"]["full_calls"][0]["track_student_grad"]
        )
        self.assertEqual(len(results["reuse_all"]["selected_calls"]), 0)
        self.assertFalse(
            results["recompute_selected"]["full_calls"][0][
                "track_student_grad"
            ]
        )
        self.assertEqual(
            len(results["recompute_selected"]["selected_calls"]),
            1,
        )

    def test_real_probe_reuse_and_recompute_selected_match_gradients(self):
        config = SimpleNamespace(
            lingbot_va_root=Path(__file__).resolve().parents[2] / "lingbot-va",
            signature_horizon=1,
            param_dtype="float32",
            frame_chunk_size=2,
            action_dim=1,
            action_per_frame=1,
            action_num_train_timesteps=10,
            action_snr_shift=1.0,
        )
        probe = ActionResponseProbe(
            _DifferentiableActionTeacher(),
            config,
            action_channel_ids=(0,),
        )
        teacher_videos = torch.tensor([1.0, 3.0]).view(
            1, 2, 1, 1, 1, 1
        ).expand(1, 2, 1, 2, 1, 1).clone()
        teacher_actions = torch.zeros(1, 2, 1, 2, 1, 1)
        action_noise = torch.zeros(1, 1, 2, 1, 1)
        timestep_ids = torch.tensor([1, 2])
        selected_teacher = torch.tensor([[0, 1]])
        condition = ActionSignatureCondition(
            text_emb=torch.zeros(1, 1, 1),
            frame_start=torch.tensor([2]),
        )

        reuse_videos = torch.tensor([2.0, 4.0]).view(
            1, 2, 1, 1, 1, 1
        ).expand(1, 2, 1, 2, 1, 1).clone().requires_grad_(True)
        reuse_batch = probe(
            teacher_videos,
            reuse_videos,
            teacher_actions,
            action_noise,
            timestep_ids,
            condition,
            track_student_grad=True,
        )
        reuse_responses = ActionExecutionLoss.select_student_responses(
            reuse_batch.student_responses,
            selected_teacher,
        )
        reuse_responses.sum().backward()

        recompute_videos = reuse_videos.detach().clone().requires_grad_(True)
        with torch.no_grad():
            detached_batch = probe(
                teacher_videos,
                recompute_videos,
                teacher_actions,
                action_noise,
                timestep_ids,
                condition,
            )
        recomputed_responses = probe.compute_selected_student_responses(
            recompute_videos,
            detached_batch.noisy_teacher_actions,
            detached_batch.action_timestep_ids,
            selected_teacher,
            condition,
        )
        recomputed_responses.sum().backward()

        torch.testing.assert_close(reuse_responses, recomputed_responses)
        torch.testing.assert_close(reuse_videos.grad, recompute_videos.grad)

    def test_bandwidth_calibrator_consumes_response_distance_matrix(self):
        features = torch.tensor([0.0, 2.0, 6.0]).view(1, 3, 1, 1, 1)
        response_distances = torch.tensor(
            [[[0.0, 1.0, 5.0], [7.0, 0.0, 3.0], [9.0, 11.0, 0.0]]]
        )

        result = BandwidthCalibrator(VideoMetric())(
            features,
            response_distances,
        )

        self.assertAlmostEqual(result.bandwidths.video, 1.0)
        self.assertAlmostEqual(result.bandwidths.action, 6.0)
        torch.testing.assert_close(
            result.action_distances.sort().values,
            torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]),
        )

    def test_calibration_helper_probes_all_teacher_videos(self):
        config = self._config()
        response_probe = _ActionResponseProbe()
        signature_cache = _TeacherSignatureCache()

        result = calibrate_bandwidths(
            config,
            self._batch(),
            response_probe,
            ActionResponseMetric(action_dim=1, action_channel_ids=(0,)),
            _VideoFeatureExtractor(),
            BandwidthCalibrator(VideoMetric()),
            signature_cache,
        )

        self.assertEqual(len(signature_cache.calls), 1)
        self.assertEqual(response_probe.calls[0]["teacher_shape"][1], 3)
        self.assertEqual(response_probe.calls[0]["student_shape"][1], 3)
        self.assertAlmostEqual(result.bandwidths.video, 0.75)
        self.assertAlmostEqual(result.bandwidths.action, 4.0)

    def test_action_response_distance_changes_positive_kernel_weights(self):
        teacher_features = torch.tensor([0.0, 10.0]).view(1, 2, 1, 1, 1)
        student_features = torch.tensor([4.0, 6.0]).view(1, 2, 1, 1, 1)
        video = VideoMetricResult(
            teacher_features=teacher_features,
            student_features=student_features,
            feature_scale=torch.tensor(1.0),
            distances=VideoDistances(
                teacher_teacher=torch.zeros(1, 2, 2),
                student_teacher=torch.zeros(1, 2, 2),
                student_student=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
            ),
        )
        action = ActionResponseMetricResult(
            student_teacher=torch.tensor([[[0.0, 4.0], [4.0, 0.0]]]),
            student_student=torch.tensor([[[0.0, 2.0], [8.0, 0.0]]]),
        )

        result = DriftingKernel()(video, action, KernelBandwidths(1.0, 1.0))

        self.assertGreater(result.positive_weights[0, 0, 0].item(), 0.98)
        self.assertGreater(result.positive_weights[0, 1, 1].item(), 0.98)
        summary = KernelDiagnostics()(
            video, action, result, KernelBandwidths(1.0, 1.0)
        )
        # Both directions count: action median=(2+8)/2; joint adds video=1.
        self.assertEqual(
            summary.scalars["distance/student_student/action_raw/median"].item(), 5.0
        )
        self.assertEqual(
            summary.scalars["distance/student_student/joint/median"].item(), 6.0
        )


if __name__ == "__main__":
    unittest.main()
