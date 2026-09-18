"""Focused contracts for the Flash-WAM-500 clean-50 training path."""

from __future__ import annotations

import random
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from train_v4.engine.resume_state import (
    capture_rng_state,
    load_resume_state,
    save_resume_state,
)
from train_v4.engine.training_data import (
    SynchronizedFrameStartSampler,
    HistoryStressSampler,
)
from train_v4.engine import teacher_signature_cache as cache_module
from train_v4.engine.teacher_signature_cache import TeacherSignatureCache
from train_v4.engine.trainer import JointDistillationTrainer
from train_v4.forwards.action_signature import (
    ActionSignatureCondition,
    ActionSignatureGenerator,
)
from train_v4.train import assert_fresh_students_identical
from train_v4.training_config import TrainingConfig


class _RandomDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return index, int(torch.randint(1_000_000, ()).item())


class _PartiallyUsedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Linear(3, 2)
        self.unused = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.used(inputs)


class _FrameStartDataset:
    def __init__(self, signatures: list[tuple[int, ...]]) -> None:
        self.signatures = signatures

    def __len__(self) -> int:
        return len(self.signatures)

    def frame_starts_for_index(self, index: int) -> tuple[int, ...]:
        return self.signatures[index]


class _FakeSignatureGenerator:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.calls = 0

    def __call__(self, videos, action_noise, condition):
        self.calls += 1
        candidate_count = videos.shape[1]
        return action_noise[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
            -1,
            -1,
        ).clone()


class _CountingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([2.0, 1.0])

    def set_timesteps(self, count: int) -> None:
        if count != 2:
            raise AssertionError("unexpected solver step count")

    @staticmethod
    def step(prediction, timestep, actions):
        del prediction, timestep
        return actions


class _CountingSignatureGenerator(ActionSignatureGenerator):
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            action_teacher_num_inference_steps=2,
            action_per_frame=1,
            action_dim=1,
        )
        self.action_scheduler = _CountingScheduler()
        self.cache_updates: list[int] = []

    def _forward(
        self,
        latents,
        text_emb,
        frame_starts,
        *,
        timestep,
        action_mode,
        update_cache,
    ):
        del text_emb, frame_starts, timestep, action_mode
        self.cache_updates.append(update_cache)
        return torch.zeros(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            latents.shape[1],
        )


class TrainingContractTest(unittest.TestCase):
    def test_main_profile_global_batch(self) -> None:
        config = TrainingConfig(
            student_init_source="flash_wam",
            world_size=4,
            batch_size=1,
            gradient_accumulation_steps=8,
            expected_world_size=4,
            expected_global_batch_size=32,
        )
        config.validate_runtime_contract()
        self.assertEqual(config.effective_global_batch_size, 32)
        self.assertEqual(
            config.selected_student_init_path.name,
            "FlashWAM-RoboTwin",
        )
        inverse = config.inverse_used_action_channel_ids
        self.assertEqual(len(inverse), config.action_dim)
        for physical_index, native_index in enumerate(
            config.used_action_channel_ids
        ):
            self.assertEqual(inverse[native_index], physical_index)
        self.assertEqual(inverse[14], len(config.used_action_channel_ids))

    def test_teacher_signature_cache_persists_signature_without_rng_effect(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig(
                student_init_source="flash_wam",
                teacher_signature_cache_path=Path(directory),
            )
            generator = _FakeSignatureGenerator(config)
            cache = object.__new__(TeacherSignatureCache)
            cache.config = config
            cache.generator = generator
            cache.allow_misses = True
            cache.root = Path(directory)
            cache.contract_sha256 = "test-contract"

            videos = torch.zeros(1, 4, 1, 2, 1, 1, dtype=torch.bfloat16)
            condition = ActionSignatureCondition(
                text_emb=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                frame_start=torch.tensor([6]),
                history_frame_start=torch.tensor([0]),
            )
            torch.manual_seed(123)
            expected_next_random = torch.rand(1)
            torch.manual_seed(123)
            first = cache.get_or_compute(("task/episode@latent_0006",), videos, condition)
            actual_next_random = torch.rand(1)

            self.assertEqual(first.hits, 0)
            self.assertEqual(first.misses, 1)
            self.assertEqual(generator.calls, 1)
            self.assertTrue(torch.equal(actual_next_random, expected_next_random))
            expected_noise = cache._deterministic_noise(
                "task/episode@latent_0006",
                (config.action_dim, 2, config.action_per_frame, 1),
            )
            self.assertTrue(torch.equal(
                first.signatures,
                expected_noise[None, None].expand_as(first.signatures),
            ))

            fresh_generator = _FakeSignatureGenerator(config)
            fresh_cache = object.__new__(TeacherSignatureCache)
            fresh_cache.config = config
            fresh_cache.generator = fresh_generator
            fresh_cache.allow_misses = True
            fresh_cache.root = Path(directory)
            fresh_cache.contract_sha256 = "test-contract"
            second = fresh_cache.get_or_compute(
                ("task/episode@latent_0006",),
                videos,
                condition,
            )

            self.assertEqual(second.hits, 1)
            self.assertEqual(second.misses, 0)
            self.assertEqual(fresh_generator.calls, 0)
            self.assertTrue(torch.equal(second.signatures, first.signatures))

    def test_formal_cache_rejects_missing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig(
                student_init_source="flash_wam",
                teacher_signature_cache_path=Path(directory),
            )
            cache = object.__new__(TeacherSignatureCache)
            cache.config = config
            cache.generator = None
            cache.allow_misses = False
            cache.root = Path(directory)
            cache.contract_sha256 = "test-contract"
            videos = torch.zeros(1, 4, 1, 2, 1, 1, dtype=torch.bfloat16)
            condition = ActionSignatureCondition(
                text_emb=torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                frame_start=torch.tensor([2]),
                history_frame_start=torch.tensor([0]),
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "does not compute 50-step teacher signatures online",
            ):
                cache.get_or_compute(("missing@latent_0002",), videos, condition)

    def test_cache_completion_marker_checks_entry_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainingConfig(
                student_init_source="flash_wam",
                teacher_signature_cache_path=root,
            )
            contract = {"test": "contract"}
            with patch.object(
                cache_module,
                "teacher_signature_cache_contract",
                return_value=contract,
            ):
                cache_module.prepare_teacher_signature_cache(config)
                entry = root / "entries" / "00" / "00" / "entry.pt"
                entry.parent.mkdir(parents=True)
                torch.save({"complete": True}, entry)
                marker = cache_module.write_teacher_signature_cache_complete(
                    config,
                    expected_entries=1,
                )
                self.assertEqual(marker["validated_entries"], 1)
                cache_module.verify_teacher_signature_cache_complete(config)
                entry.unlink()
                with self.assertRaisesRegex(ValueError, "has 0 entries"):
                    cache_module.verify_teacher_signature_cache_complete(config)

    def test_cache_source_compatibility_is_limited_to_validated_exact_hash(self):
        from train_v4.engine.inference_compatibility import (
            INFERENCE_COMPATIBLE_SOURCE_HASHES,
            inference_contract_source_hash,
        )
        source = Path(__file__).resolve().parents[2] / 'lingbot-va/wan_va/modules/model.py'
        content = source.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        self.assertIn(actual, INFERENCE_COMPATIBLE_SOURCE_HASHES)
        self.assertEqual(inference_contract_source_hash(actual),
                         INFERENCE_COMPATIBLE_SOURCE_HASHES[actual])
        changed = hashlib.sha256(content + b'\n# unrelated source edit\n').hexdigest()
        self.assertEqual(inference_contract_source_hash(changed), changed)

    def test_cache_contract_allows_storage_relocation_only(self) -> None:
        stored_contract = {
            "dataset_path": "/shared/dataset",
            "teacher_video_bank_path": "/shared/bank",
            "teacher_video_bank_marker_sha256": "bank-content",
            "solver_steps": 50,
        }
        payload = {
            "contract": stored_contract,
            "contract_sha256": cache_module._contract_digest(stored_contract),
        }
        relocated = dict(stored_contract)
        relocated["dataset_path"] = "/local/dataset"
        relocated["teacher_video_bank_path"] = "/local/bank"

        validated = cache_module._validate_contract_payload(payload, relocated)
        self.assertEqual(
            validated["contract_sha256"],
            payload["contract_sha256"],
        )
        changed_content = dict(relocated)
        changed_content["solver_steps"] = 4
        with self.assertRaisesRegex(ValueError, "differs from this run"):
            cache_module._validate_contract_payload(payload, changed_content)

    def test_online_and_ema_identity_assertion(self) -> None:
        online = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        target.load_state_dict(online.state_dict())
        assert_fresh_students_identical(online, target)
        with torch.no_grad():
            target.weight[0, 0].add_(1)
        with self.assertRaisesRegex(RuntimeError, "not identical"):
            assert_fresh_students_identical(online, target)

    def test_action_signature_accepts_five_dimensional_noise(self) -> None:
        config = TrainingConfig(student_init_source="flash_wam")
        generator = object.__new__(ActionSignatureGenerator)
        generator.config = config
        frame_count = config.signature_horizon * config.frame_chunk_size
        videos = torch.zeros(1, 2, 48, frame_count, 24, 20)
        action_noise = torch.zeros(
            1,
            config.action_dim,
            frame_count,
            config.action_per_frame,
            1,
        )
        condition = ActionSignatureCondition(text_emb=torch.zeros(1, 1, 1))

        generator._validate_inputs(videos, action_noise, condition)
        with self.assertRaisesRegex(ValueError, "action_noise must have shape"):
            generator._validate_inputs(
                videos,
                action_noise.unsqueeze(1),
                condition,
            )

    def test_action_signature_skips_unused_final_cache_forward(self) -> None:
        generator = _CountingSignatureGenerator()
        args = (
            torch.zeros(1, 1, 2, 1, 1),
            torch.zeros(1, 1, 1),
            torch.tensor([2]),
            torch.tensor([False]),
            torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
        )

        generator._denoise_action_chunk(
            *args,
            cache_clean_action=False,
        )
        self.assertEqual(generator.cache_updates, [0, 0])

        generator.cache_updates.clear()
        generator._denoise_action_chunk(
            *args,
            cache_clean_action=True,
        )
        self.assertEqual(generator.cache_updates, [0, 0, 1])

    def test_sampler_synchronizes_frame_start_across_ranks(self) -> None:
        signatures = (
            [(0, 2, 4)] * 7
            + [(0, 2)] * 5
            + [(0, 2, 4, 6)] * 7
        )
        dataset = _FrameStartDataset(signatures)

        def collect() -> list[list[tuple[int, int]]]:
            return [
                list(
                    SynchronizedFrameStartSampler(
                        dataset,
                        num_replicas=3,
                        rank=rank,
                        batch_size=2,
                        seed=42,
                    )
                )
                for rank in range(3)
            ]

        streams = collect()
        self.assertEqual(streams, collect())
        for offset in range(0, len(streams[0]), 2):
            global_batch = [
                item
                for rank_stream in streams
                for item in rank_stream[offset : offset + 2]
            ]
            self.assertEqual(len({start for _, start in global_batch}), 1)
            self.assertEqual(len({source for source, _ in global_batch}), 6)

    def test_history_stress_order_is_valid_and_synchronized(self) -> None:
        # Rare longest chunks repeat across ranks in this diagnostic only.
        dataset = _FrameStartDataset(
            [(0, 24, 50)] * 12 + [(0, 24)] * 8 + [(0, 80)] * 3
        )
        for updates in (1, 2, 3):
            streams = []
            for rank in range(4):
                config = TrainingConfig(
                    world_size=4, rank=rank, batch_size=1,
                    gradient_accumulation_steps=8, max_train_steps=updates,
                    history_stress_test=True, save_final_checkpoint=False,
                )
                config.validate_runtime_contract()
                sampler = HistoryStressSampler(dataset, config=config)
                stream = list(sampler)
                self.assertEqual(stream, list(sampler))
                self.assertEqual(len(stream), len(sampler))
                expected = ([80] + [24] * 7 + [24] * 7 + [80] + [0] * 8)
                self.assertEqual([start for _, start in stream], expected[:updates * 8])
                streams.append(stream)
            for global_batch in zip(*streams):
                self.assertEqual(len({start for _, start in global_batch}), 1)
                expected_sources = 3 if global_batch[0][1] == 80 else 4
                self.assertEqual(len({source for source, _ in global_batch}), expected_sources)
                for source, start in global_batch:
                    self.assertIn(start, dataset.frame_starts_for_index(source))

    def test_history_stress_requires_available_boundary(self) -> None:
        config = TrainingConfig(
            world_size=4, history_stress_test=True, max_train_steps=2,
            save_final_checkpoint=False,
        )
        for signatures in ([(0, 50)] * 8, [(0, 24)] * 8):
            with self.assertRaisesRegex(ValueError, "sync boundary"):
                HistoryStressSampler(_FrameStartDataset(signatures), config=config)

    def test_history_stress_cannot_be_resumed_or_saved_as_training(self) -> None:
        from dataclasses import replace
        config = TrainingConfig(
            history_stress_test=True, max_train_steps=2,
            save_final_checkpoint=False,
        )
        config.validate_runtime_contract()
        for overrides in (
            {"max_train_steps": 4}, {"save_final_checkpoint": True},
            {"save_interval": 1}, {"resume_from_step": 1},
            {"resume_from_path": Path("/tmp/old-run")},
        ):
            with self.assertRaises(ValueError):
                replace(config, **overrides).validate_runtime_contract()

    def test_optimizer_and_rng_round_trip(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(4, 3)).square().mean().backward()
        optimizer.step()
        expected = optimizer.state_dict()
        loader = DataLoader(
            TensorDataset(torch.arange(4)),
            batch_size=1,
            generator=torch.Generator().manual_seed(7),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_resume_state(
                model,
                optimizer,
                checkpoint,
                capture_rng_state(torch.device("cpu"), loader),
            )
            restored = torch.optim.AdamW(model.parameters(), lr=9e-3)
            rng_state = load_resume_state(model, restored, checkpoint)

        actual = restored.state_dict()
        self.assertTrue(actual["state"])
        self.assertEqual(
            actual["param_groups"][0]["lr"],
            expected["param_groups"][0]["lr"],
        )
        self.assertTrue(
            torch.equal(
                actual["state"][0]["exp_avg"],
                expected["state"][0]["exp_avg"],
            )
        )
        self.assertIn("torch_cpu", rng_state)

    def test_optimizer_resume_allows_never_used_parameters(self) -> None:
        model = _PartiallyUsedModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(4, 3)).square().mean().backward()
        optimizer.step()
        self.assertEqual(len(optimizer.state), 2)
        loader = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_resume_state(
                model,
                optimizer,
                checkpoint,
                capture_rng_state(torch.device("cpu"), loader),
            )
            restored = torch.optim.AdamW(model.parameters(), lr=9e-3)
            load_resume_state(model, restored, checkpoint)

        actual = restored.state_dict()
        self.assertEqual(len(actual["state"]), 4)
        self.assertEqual(actual["state"][2]["step"].item(), 0)
        self.assertEqual(torch.count_nonzero(actual["state"][2]["exp_avg"]).item(), 0)

    def test_resumed_loader_replays_position_then_restores_rng(self) -> None:
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader = DataLoader(_RandomDataset(), batch_size=1, shuffle=False)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        saved_rng = capture_rng_state(torch.device("cpu"), loader)
        expected = next(iterator)

        resumed_loader = DataLoader(_RandomDataset(), batch_size=1, shuffle=False)
        trainer = object.__new__(JointDistillationTrainer)
        trainer.train_loader = resumed_loader
        trainer.epoch = 0
        trainer.batches_consumed_in_epoch = 2
        trainer._loader_iterator = None
        trainer._pending_rng_state = saved_rng
        trainer.device = torch.device("cpu")

        actual = JointDistillationTrainer._next_batch(trainer)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(trainer.batches_consumed_in_epoch, 3)


if __name__ == "__main__":
    unittest.main()
