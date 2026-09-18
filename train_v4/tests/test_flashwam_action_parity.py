"""Compare v4 values/gradients with pinned upstream loss on corrected native attention.

Uses the official DataMixin and StepMixin source, without rewriting their loss.
Covers full sequences, including first-frame loss and a partial final chunk.
"""
import ast, copy, subprocess, types, logging, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from einops import rearrange
from train_v4.tests.test_native_attention import load_native
from train_v4.objectives.action_consistency_loss import ActionConsistencyLoss
from train_v4.objectives.action_training_pair import ActionTrainingPair
from train_v4.forwards.action_signature import ActionSignatureCondition
from train_v4.config import DistillationConfig
from train_v4.runtime import activate_lingbot_va
activate_lingbot_va(Path(__file__).resolve().parents[2] / "lingbot-va")
from wan_va.utils import FlowMatchScheduler, get_mesh_id
ROOT=Path(__file__).resolve().parents[2]
REV='5b8df13e9db24fb15ce42ff5ccc60a4015195960'
def original_class(path, name, symbols):
    source=subprocess.check_output(['git','-C',str(ROOT/'Flash-WAM'),'show',f'{REV}:{path}'],text=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),f'official/{path}','exec'),symbols)
    return symbols[name]
ids=torch.tensor([100,750])
DataMixin=original_class('distillation/data.py','DataMixin',dict(torch=torch,get_mesh_id=get_mesh_id,sample_timestep_id=lambda **kw:torch.tensor([100,750,0,499,999])[:kw['batch_size']]))
StepMixin=original_class('distillation/step.py','StepMixin',dict(torch=torch,F=torch.nn.functional,rearrange=rearrange,logger=logging.getLogger('reference')))
module=load_native('lingbot-va/wan_va/modules/model.py','parity_native')
class Reference(DataMixin,StepMixin):
    def _prepare_input_dict(self,batch): return copy.deepcopy(self.prepared)
    def _extract_video_v(self,output,shape,batch): return torch.zeros(shape,dtype=output.dtype,device=output.device)
class OfficialActionParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_fp32_and_bf16_values_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.compare(dtype, torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_bf16_values_and_gradients(self):
        self.compare(torch.bfloat16, torch.device('cuda'))

    def compare(self, dtype, device):
        torch.manual_seed(67)
        frames=5
        ids=torch.tensor([100,750,0,499,999])
        config=DistillationConfig(param_dtype=str(dtype).split('.')[-1])
        def model():
            return module.WanTransformer3DModel(patch_size=[1,1,1],num_attention_heads=1,attention_head_dim=24,in_channels=4,out_channels=4,action_dim=30,text_dim=8,freq_dim=16,ffn_dim=32,num_layers=2,attn_mode='torch').to(device=device,dtype=dtype).eval()
        teacher=model().requires_grad_(False); target=model().requires_grad_(False); online=model()
        ref_online=copy.deepcopy(online)
        ref_online.set_requires_gradient_sync=lambda _:None
        obj=ActionConsistencyLoss(teacher,online,target,config,action_channel_ids=range(16))
        clean=torch.randn(1,30,frames,16,1,device=device)*.5
        clean[:,0,0,0,0]=3.25
        clean[:,1,0,0,0]=-2.0
        mask=torch.zeros_like(clean,dtype=torch.bool);mask[:,:16]=True
        clean.masked_fill_(~mask,0)
        video=torch.randn(1,4,frames,1,1,dtype=dtype,device=device)
        pair=ActionTrainingPair(video,clean,mask)
        condition=ActionSignatureCondition(text_emb=torch.randn(1,5,8,dtype=dtype,device=device),frame_start=torch.tensor([0]),history_frame_start=torch.tensor([0]))
        packed=obj._pack_condition(video,clean,condition,device=device,dtype=dtype)
        ref=Reference();ref.device=device;ref.patch_size=(1,1,1)
        torch.manual_seed(456)
        action=DataMixin._add_noise(ref,clean.clone(),obj.action_scheduler,action_mask=mask,action_mode=True)
        prepared=obj._model_input(packed,action['noisy_latents'],action['timesteps'])
        prepared['action_dict'].update(targets=action['targets'],actions_mask=mask)
        # The official video trajectory is irrelevant to legal action queries.
        prepared['latent_dict']['noisy_latents']=torch.randn_like(video)
        prepared['latent_dict']['timesteps']=action['timesteps'].clone()
        ref.prepared=prepared
        ref.config=types.SimpleNamespace(num_train_timesteps=1000,cfg_min=2.,cfg_max=10.,loss_type='huber',huber_c=.001,action_loss_weight=1.,action_aware_weight=.01,rank=0)
        ref.teacher=teacher;ref.student=ref_online;ref.target_student=target
        ref.distill_action=True;ref.distill_video=False;ref.action_aware=True;ref.action_distill_mode='x0';ref.k=500;ref.k_action=500
        ref.gradient_accumulation_steps=1;ref.train_scheduler_latent=obj.action_scheduler;ref.train_scheduler_action=obj.action_scheduler
        ref.empty_emb=torch.zeros_like(condition.text_emb);ref.step=0
        expected=ref._train_step(dict(latents=video,actions=clean,actions_mask=mask),0)
        torch.manual_seed(456)
        with patch('torch.randint',return_value=ids):
            actual=obj(pair,condition)
        (actual.consistency_loss+.01*actual.flow_matching_loss).backward()
        torch.testing.assert_close(actual.consistency_loss,expected['action_loss'],rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(actual.flow_matching_loss,expected['action_aware_loss'],rtol=1e-6,atol=1e-8)
        for p,q in zip(online.parameters(),ref_online.parameters()):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7)

if __name__ == '__main__':
    unittest.main()
