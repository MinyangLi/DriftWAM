"""Checkpoint capacity, cleanup, resume, and logging contracts."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from train_v4.engine.trainer import JointDistillationTrainer
from train_v4.engine.checkpoint_retention import prune_old_checkpoints
from train_v4.engine.precision import TRAINING_NUMERICS
from train_v4.run_manifest import (
    CHECKPOINT_DISK_HEADROOM_BYTES,
    estimate_checkpoint_storage,
    _validate_output_location,
)
from train_v4.training_config import TrainingConfig


class CheckpointSafetyTest(unittest.TestCase):
    def test_storage_estimate_bounds_five_saves_to_one_full_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = Path(directory) / "transformer"
            transformer.mkdir()
            save_file(
                {"weight": torch.zeros(5, dtype=torch.bfloat16)},
                transformer / "weights.safetensors",
            )
            config = TrainingConfig(
                student_init_source="flash_wam",
                max_train_steps=250,
                save_interval=50,
                retain_checkpoint_count=1,
                save_optimizer_state=True,
            )

            estimate = estimate_checkpoint_storage(config, transformer, 0)

        self.assertEqual(estimate["checkpoint_steps"], (50, 100, 150, 200, 250))
        self.assertEqual(estimate["model_pair_bytes"], 40)
        self.assertEqual(estimate["resume_state_estimate_bytes"], 40)
        self.assertEqual(estimate["peak_additional_bytes"], 80)
        self.assertEqual(
            estimate["required_free_bytes"],
            80 + CHECKPOINT_DISK_HEADROOM_BYTES,
        )

    def test_storage_estimate_for_resume_only_counts_future_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transformer = Path(directory) / "transformer"
            transformer.mkdir()
            save_file(
                {"weight": torch.zeros(5, dtype=torch.bfloat16)},
                transformer / "weights.safetensors",
            )
            config = TrainingConfig(
                student_init_source="flash_wam",
                max_train_steps=250,
                save_interval=125,
                retain_checkpoint_count=1,
                save_optimizer_state=True,
            )

            estimate = estimate_checkpoint_storage(config, transformer, 125)

        self.assertEqual(estimate["checkpoint_steps"], (250,))
        self.assertEqual(estimate["peak_additional_bytes"], 80)

    def test_failed_model_save_removes_unpublished_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            previous = checkpoint_dir / "step_0"
            previous.mkdir()
            (previous / "CHECKPOINT_COMPLETE").write_text("{}")
            (previous / "weights").write_bytes(b"previous weights")
            trainer = object.__new__(JointDistillationTrainer)
            trainer.checkpoint_dir = checkpoint_dir
            trainer.step = 1
            trainer.student = object()
            trainer.target_student = object()
            trainer.config = SimpleNamespace(
                rank=0,
                save_optimizer_state=False,
                retain_checkpoint_count=1,
            )

            def fail_after_partial_write(model, destination):
                del model
                destination.mkdir(parents=True)
                (destination / "partial.safetensors").write_bytes(b"partial")
                raise RuntimeError("simulated save failure")

            trainer._save_model = fail_after_partial_write
            with self.assertRaisesRegex(RuntimeError, "simulated save failure"):
                trainer.save_checkpoint()

            # Delete-before-save deliberately provides no old fallback.
            self.assertEqual(list(checkpoint_dir.iterdir()), [])
            self.assertFalse(previous.exists())


    @staticmethod
    def _complete_checkpoint(root, step):
        path = root / f"step_{step}"
        path.mkdir(parents=True)
        for name in ("CHECKPOINT_COMPLETE", "RESUME_STATE_COMPLETE"):
            (path / name).write_text("{}")
        for name in ("online_student", "target_student", "resume_state"):
            (path / name).mkdir()
            (path / name / "weights").write_bytes(b"complete payload")
        return path

    def _fake_save_trainer(self, root):
        trainer = object.__new__(JointDistillationTrainer)
        trainer.config = TrainingConfig(
            student_init_source="flash_wam", output_dir=root,
            rank=0, world_size=1, max_train_steps=250,
        )
        trainer.checkpoint_dir = root / "checkpoints"
        trainer.checkpoint_dir.mkdir(exist_ok=True)
        trainer.step = 100
        trainer.epoch = 0
        trainer.batches_consumed_in_epoch = 8
        trainer.device = torch.device("cpu")
        trainer.student = trainer.target_student = trainer.optimizer = object()
        trainer.train_loader = object()
        trainer.training_step = SimpleNamespace(
            bandwidths=SimpleNamespace(state_dict=lambda: {}),
        )

        def save_model(model, destination):
            destination.mkdir(parents=True)
            (destination / "weights").write_bytes(b"model payload")

        def save_resume(model, optimizer, destination, rng):
            (destination / "resume_state").mkdir()
            (destination / "resume_state" / "weights").write_bytes(b"optimizer payload")

        trainer._save_model = save_model
        for name, kwargs in (
            ("capture_rng_state", {"return_value": {}}),
            ("save_resume_state", {"side_effect": save_resume}),
        ):
            patcher = patch(f"train_v4.engine.trainer.{name}", **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        return trainer

    def test_delete_before_save_preserves_retained_and_unrelated_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self._complete_checkpoint(root, 50)
            middle = self._complete_checkpoint(root, 100)
            current = root / "step_150"
            future = self._complete_checkpoint(root, 250)
            incomplete = root / "step_75"
            incomplete.mkdir()
            unrelated = root / "notes"
            unrelated.mkdir()
            symlink = root / "step_25"
            symlink.symlink_to(future, target_is_directory=True)
            # Reserve one of two retained slots for the new save.
            prune_old_checkpoints(current, 2)
            self.assertFalse(old.exists())
            self.assertTrue(middle.is_dir())
            prune_old_checkpoints(current, 1)
            self.assertFalse(middle.exists())
            self.assertFalse(current.exists())
            for path in (future, incomplete, unrelated, symlink):
                self.assertTrue(path.exists(), path)

    def test_new_save_waits_for_deletion_to_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._fake_save_trainer(Path(directory))
            self._complete_checkpoint(trainer.checkpoint_dir, 50)
            entered, release, model_write_started = Event(), Event(), Event()
            real_rmtree = shutil.rmtree
            real_save_model = trainer._save_model

            def slow_rmtree(path, *args, **kwargs):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release checkpoint deletion")
                return real_rmtree(path, *args, **kwargs)

            def observed_save_model(*args):
                model_write_started.set()
                return real_save_model(*args)

            trainer._save_model = observed_save_model
            with patch("train_v4.engine.checkpoint_retention.shutil.rmtree", side_effect=slow_rmtree):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(trainer.save_checkpoint)
                    try:
                        self.assertTrue(entered.wait(5))
                        self.assertFalse(pending.done())
                        self.assertFalse(model_write_started.is_set())
                        self.assertTrue((trainer.checkpoint_dir / ".step_50.deleting").is_dir())
                        self.assertFalse(list(trainer.checkpoint_dir.glob(".step_100.*.tmp")))
                    finally:
                        release.set()
                    latest = pending.result(timeout=10)
            self.assertTrue(model_write_started.is_set())
            self.assertTrue((latest / "RESUME_STATE_COMPLETE").is_file())
            self.assertEqual(list(trainer.checkpoint_dir.iterdir()), [latest])

    def test_failed_deletion_prevents_new_write_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._fake_save_trainer(Path(directory))
            self._complete_checkpoint(trainer.checkpoint_dir, 50)
            with patch("train_v4.engine.checkpoint_retention.shutil.rmtree", side_effect=OSError("deletion denied")):
                with self.assertRaisesRegex(RuntimeError, "failed to delete old checkpoints"):
                    trainer.save_checkpoint()
            self.assertFalse((trainer.checkpoint_dir / "step_100").exists())
            self.assertFalse(list(trainer.checkpoint_dir.glob(".step_100.*.tmp")))
            self.assertTrue((trainer.checkpoint_dir / ".step_50.deleting").is_dir())
            # Retrying reclaims a partially deleted old directory before writing.
            latest = trainer.save_checkpoint()
            self.assertEqual(list(trainer.checkpoint_dir.iterdir()), [latest])

    def test_duplicate_save_is_rejected_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._fake_save_trainer(Path(directory))
            old = self._complete_checkpoint(trainer.checkpoint_dir, 50)
            current = self._complete_checkpoint(trainer.checkpoint_dir, 100)
            with self.assertRaisesRegex(RuntimeError, "checkpoint already exists"):
                trainer.save_checkpoint()
            self.assertTrue((old / "RESUME_STATE_COMPLETE").is_file())
            self.assertTrue((current / "RESUME_STATE_COMPLETE").is_file())

    def test_resume_capacity_credits_checkpoint_deleted_before_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer = root / "transformer"
            transformer.mkdir()
            save_file({"weight": torch.zeros(5, dtype=torch.bfloat16)}, transformer / "weights.safetensors")
            old = self._complete_checkpoint(root / "checkpoints", 50)
            (old / "payload").write_bytes(b"x" * 80)
            config = TrainingConfig(
                student_init_source="flash_wam", output_dir=root,
                max_train_steps=250, save_interval=50,
            )
            estimate = estimate_checkpoint_storage(config, transformer, 50)
            self.assertGreaterEqual(estimate["reclaimable_existing_checkpoint_bytes"], 80)
            self.assertEqual(estimate["peak_additional_bytes"], 0)
            self.assertEqual(estimate["required_free_bytes"], CHECKPOINT_DISK_HEADROOM_BYTES)

    def test_failed_completion_marker_leaves_no_old_fallback(self):
        from train_v4.engine import trainer as module
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._fake_save_trainer(Path(directory))
            old = self._complete_checkpoint(trainer.checkpoint_dir, 50)
            dump = module._atomic_json_dump

            def fail_resume_marker(value, path):
                if path.name == "RESUME_STATE_COMPLETE":
                    raise OSError("marker write failed")
                return dump(value, path)

            with patch.object(module, "_atomic_json_dump", side_effect=fail_resume_marker):
                with self.assertRaisesRegex(RuntimeError, "marker write failed"):
                    trainer.save_checkpoint()
            self.assertFalse(old.exists())
            self.assertFalse((trainer.checkpoint_dir / "step_100" / "RESUME_STATE_COMPLETE").exists())

    def test_resume_rejects_changed_action_weight(self) -> None:
        config = TrainingConfig(
            student_init_source="flash_wam",
            action_consistency_loss_weight=1.0,
        )
        state = {
            "training_numerics": TRAINING_NUMERICS,
            "param_dtype": config.param_dtype,
            "action_supervision_source": config.action_supervision_source,
            "step": 125,
            "world_size": config.world_size,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "seed": config.seed,
            "execution_response_mode": config.execution_response_mode,
            "execution_loss_weight": config.execution_loss_weight,
            "action_consistency_loss_weight": 0.5,
            "action_flow_matching_loss_weight": (
                config.action_flow_matching_loss_weight
            ),
            "video_drifting_loss_weight": config.video_drifting_loss_weight,
            "beta_rep": config.beta_rep,
            "ema_decay": config.ema_decay,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "weight_decay": config.weight_decay,
            "optimizer_epsilon": config.optimizer_epsilon,
            "max_grad_norm": config.max_grad_norm,
            "teacher_signature_noise_seed": (
                config.teacher_signature_noise_seed
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
            trainer = object.__new__(JointDistillationTrainer)
            trainer.config = config
            trainer.step = 125

            with self.assertRaisesRegex(
                ValueError,
                "action_consistency_loss_weight",
            ):
                trainer._restore_training_state(checkpoint)

    def test_resume_rejects_teacher_pair_or_missing_source_before_loading(self):
        from dataclasses import asdict
        config = TrainingConfig(student_init_source="flash_wam")
        state = asdict(config)
        state.update(step=125, training_numerics=TRAINING_NUMERICS,
                     epoch=0, batches_consumed_in_epoch=0)
        with tempfile.TemporaryDirectory() as directory:
            config.output_dir = Path(directory)
            checkpoint = config.output_dir / "checkpoints" / "step_125"
            checkpoint.mkdir(parents=True)
            (checkpoint / "CHECKPOINT_COMPLETE").touch()
            (checkpoint / "RESUME_STATE_COMPLETE").touch()
            trainer = object.__new__(JointDistillationTrainer)
            trainer.config = config
            trainer.step = 125
            trainer.train_loader = [None]
            trainer.student = object()
            trainer.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
            for source in (None, "teacher_generated", "ground_truth"):
                if source is None:
                    state.pop("action_supervision_source", None)
                else:
                    state["action_supervision_source"] = source
                (checkpoint / "trainer_state.json").write_text(json.dumps(state, default=str))
                if source != "ground_truth":
                    with self.assertRaisesRegex(ValueError, "action_supervision_source"):
                        _validate_output_location(config, checkpoint)
                    with self.assertRaisesRegex(ValueError, "action_supervision_source"):
                        trainer._restore_training_state(checkpoint)
                else:
                    _validate_output_location(config, checkpoint)
                    with patch("train_v4.engine.trainer.load_resume_state", return_value={}) as restore:
                        trainer._restore_training_state(checkpoint)
                    restore.assert_called_once()

    def test_log_persists_and_prints_action_losses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(JointDistillationTrainer)
            trainer.config = SimpleNamespace(
                rank=0,
                output_dir=Path(directory),
            )
            trainer.step = 10
            trainer._wandb_run = None
            values = {
                "loss/total": 3.0,
                "loss/video_drifting_contribution": 1.0,
                "loss/video_execution_contribution": 1.0,
                "loss/action_consistency": 0.75,
                "loss/action_flow_matching": 0.25,
                "gradient/total_norm_before_clip": 2.0,
                "gradient/clipping_factor": 1.0,
                "train/learning_rate": 2e-6,
                "cache/teacher_signature_hit_rate": 1.0,
            }

            with self.assertLogs(
                "train_v4.engine.trainer",
                level="INFO",
            ) as captured:
                trainer._log(values)

            log_text = "\n".join(captured.output)
            self.assertIn("action_consistency=0.750000", log_text)
            self.assertIn("action_flow=0.250000", log_text)
            record = json.loads(
                (Path(directory) / "metrics.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["loss/action_consistency"], 0.75)
            self.assertEqual(record["loss/action_flow_matching"], 0.25)

    def test_logging_failure_is_fatal(self) -> None:
        trainer = object.__new__(JointDistillationTrainer)
        trainer.config = SimpleNamespace(rank=0)
        trainer.step = 10
        trainer._wandb_run = None
        with patch.object(
            trainer,
            "_append_metrics_jsonl",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "failed to persist training diagnostics",
            ):
                trainer._log({"train/learning_rate": 2e-6})


if __name__ == "__main__":
    unittest.main()
