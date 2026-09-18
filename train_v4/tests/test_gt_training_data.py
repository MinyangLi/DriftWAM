"""Full GT supervision with independent causal video slicing through the actual dataset/collation pipeline."""
import ast
import subprocess
import tempfile
import numpy as np
from einops import rearrange
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_v4.engine.training_data import DriftingTrainingDataset, collate_training_samples


class _SourceDataset(list):
    pass


class _MultiDataset:
    def __init__(self, child):
        self._datasets = [child]
        self.item_id_to_dataset_id = {0: 0}
        self.acc_dset_num = [0]

    def __len__(self):
        return 1


class GroundTruthTrainingDataTest(unittest.TestCase):
    def test_current_gt_slicing_history_masks_and_device_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                'latents': torch.arange(12, dtype=torch.float32).reshape(2, 6, 1, 1),
                'actions': torch.arange(30*6*16, dtype=torch.float32).reshape(30, 6, 16, 1),
                'actions_mask': torch.ones(30, 6, 16, 1, dtype=torch.bool),
                'text_emb': torch.zeros(3, 8),
            }
            source['actions_mask'][3, 3, 7] = False
            child = _SourceDataset([source])
            child.repo_id = str(root/'task')
            child.new_metas = [dict(episode_index=0, start_frame=0, end_frame=95)]
            dataset = object.__new__(DriftingTrainingDataset)
            dataset.config = SimpleNamespace(dataset_path=root, teacher_video_bank_path=root/'bank',
                                             action_dim=30, action_per_frame=16, frame_chunk_size=2)
            dataset.base_dataset = _MultiDataset(child)
            bank = dict(teacher_videos=torch.full((3, 2, 2, 2, 1, 1), -11.),
                        frame_starts=torch.tensor([0, 2, 4]))
            with patch.object(dataset, '_load_bank', return_value=bank):
                initial = dataset[(0, 0)]
                middle = dataset[(0, 2)]
            torch.testing.assert_close(middle.gt_video, source['latents'])
            torch.testing.assert_close(middle.gt_action, source['actions'])
            torch.testing.assert_close(middle.history_video, source['latents'][:, :2])
            torch.testing.assert_close(middle.history_action, source['actions'][:, :2].clamp(-1.5,1.5))
            torch.testing.assert_close(middle.action_valid_mask, source['actions_mask'][:, 2:4])
            self.assertTrue(middle.teacher_videos.eq(-11).all())
            self.assertIsNone(initial.history_video)
            self.assertFalse(initial.action_valid_mask[:, 0].any())
            torch.testing.assert_close(initial.gt_video, source['latents'])
            self.assertTrue(source['actions_mask'][:, 0].all())
            self.assertTrue(initial.gt_action_mask[:, 0].all())
            self.assertTrue(middle.gt_action.abs().gt(1.5).any())
            torch.testing.assert_close(middle.gt_action_mask, source['actions_mask'])
            batch = collate_training_samples([middle, middle]).to('cpu', dtype=torch.bfloat16)
            torch.testing.assert_close(batch.gt_video, middle.gt_video[None].expand(2,-1,-1,-1,-1).bfloat16())
            torch.testing.assert_close(batch.gt_action, middle.gt_action[None].expand(2,-1,-1,-1,-1).float())
            self.assertEqual(batch.action_valid_mask.dtype, torch.bool)
            self.assertEqual(batch.frame_start.dtype, torch.long)
            condition = batch.condition()
            self.assertFalse(hasattr(condition, 'gt_video'))
            self.assertFalse(hasattr(condition, 'gt_action'))
            torch.testing.assert_close(condition.history_video, batch.history_video)


    def test_unclipped_native_action_preprocessing_matches_upstream(self):
        root=Path(__file__).resolve().parents[2]
        relative='wan_va/dataset/lerobot_latent_dataset.py'
        upstream=subprocess.check_output([
            'git','-C',str(root/'Flash-WAM'),'show',
            '5b8df13e9db24fb15ce42ff5ccc60a4015195960:'+relative,
        ],text=True)
        def function(source):
            cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='LatentLeRobotDataset')
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_action_post_process')
            symbols=dict(np=np,torch=torch,rearrange=rearrange)
            exec(compile(ast.Module(body=[method],type_ignores=[]),'<action_preprocessing>','exec'),symbols)
            return symbols['_action_post_process']
        # Isolate alignment/quantile normalization; pose conversion is unchanged.
        obj=SimpleNamespace(config=SimpleNamespace(env_type='normalization_test',inverse_used_action_channel_ids=list(range(17))),
                            q01=np.full(17,-.2),q99=np.full(17,.8),clip_normalized_actions=False)
        actions=np.linspace(-3,3,8*16).reshape(8,16)
        ids=np.arange(5)
        expected=function(upstream)(obj,0,8,ids,actions.copy())
        for folder in ('lingbot-va','Flash-WAM'):
            actual=function((root/folder/relative).read_text())(obj,0,8,ids,actions.copy())
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(actual[0].abs().gt(1.5).any())
            self.assertTrue(actual[1][:16,0].all())
            self.assertGreater(actual[0][:16,0].abs().max().item(),0)


if __name__ == '__main__':
    unittest.main()
