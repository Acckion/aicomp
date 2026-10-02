"""Frozen correspondence pretraining drives local IR feature resampling.
RGB/IR projections receive visual content, never absolute position embeddings.
The local candidate bound is a geometric prior, not an IR box label.
"""
from pathlib import Path
import train_baseline as baseline
import torch
import torch.nn.functional as F
from src.core import register
from ir_content_alignment import IRContentDFINE
from ir_corr_pretrain import Correspondence, tokens
@register()
class IRCorrespondenceDFINE(IRContentDFINE):
 def __init__(self,backbone,encoder,decoder,ir_enabled=True,correspondence_enabled=True,correspondence_checkpoint='/home/fbohan/AIC/experiments/ir_corr/weights_epoch_004.pth'):
  super().__init__(backbone,encoder,decoder,ir_enabled=ir_enabled)
  self.ir_corr=Correspondence(encoder.hidden_dim);state=torch.load(correspondence_checkpoint,map_location='cpu',weights_only=False);self.ir_corr.load_state_dict(state['model'],strict=True);self.ir_corr.requires_grad_(False)
  self.ir_corr_strength=torch.nn.Parameter(torch.zeros(()));self.correspondence_enabled=correspondence_enabled;self.encoder.register_forward_hook(self._align)
 def _align(self,module,inputs,output):
  if not self.ir_enabled or not self.correspondence_enabled or self._feature is None:return
  with torch.no_grad(),torch.autocast(device_type=self._feature.device.type,enabled=False):
   rgb=F.adaptive_avg_pool2d(output[0].detach().float(),(16,16));ir=F.adaptive_avg_pool2d(self._feature.float(),(16,16));a=tokens(self.ir_corr.encode(rgb,'rgb'));b=tokens(self.ir_corr.encode(ir,'ir'))
   axes=(torch.arange(16,device=a.device)+.5)/16;yy,xx=torch.meshgrid(axes,axes,indexing='ij');coords=torch.stack([xx,yy],-1).reshape(256,2)
   locality=(coords[:,None]-coords[None]).abs().amax(-1)<=.20
   valid=F.interpolate(self._valid.float(),size=(16,16),mode='nearest').flatten(2)[:,0]>.5
   score=(a@b.transpose(1,2))/.1;allow=locality[None]&valid[:,None];prob=score.masked_fill(~allow,-1e4).softmax(-1)*allow;prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-6)
   expected=prob@coords;offset=expected-coords[None]
   # Confidence arises from concentration; ambiguous neighborhoods retain
   # original positions rather than applying unconstrained maximum matching.
   entropy=-(prob*prob.clamp_min(1e-8).log()).sum(-1);maxentropy=allow.sum(-1).clamp_min(2).float().log();confidence=(1-entropy/maxentropy).clamp(0,1)
   offset=offset*confidence[...,None]
   delta=F.interpolate(offset.reshape(-1,16,16,2).permute(0,3,1,2),size=self._feature.shape[-2:],mode='bilinear',align_corners=False)
   h,w=self._feature.shape[-2:];gy,gx=torch.meshgrid((torch.arange(h,device=a.device)+.5)/h,(torch.arange(w,device=a.device)+.5)/w,indexing='ij');grid=torch.stack([gx,gy],-1)[None]+delta.permute(0,2,3,1)
   warped=F.grid_sample(self._feature.float(),grid*2-1,align_corners=False).to(self._feature.dtype)
  # New route starts at exact original IR path and retains that residual.
  self._feature=self._feature+self.ir_corr_strength.tanh()*(warped-self._feature)

_original=baseline.BaselineSolver.load_tuning_state
def load_tuning(self,path):
 if not isinstance(self.model,IRCorrespondenceDFINE):return _original(self,path)
 ck=torch.load(path,map_location='cpu',weights_only=False);weights=dict(ck['ema']['module'] if 'ema' in ck else ck['model']);current=self.model.state_dict()
 for key in ('decoder.anchors','decoder.valid_mask'):
  if key in current:weights[key]=current[key]
 result=self.model.load_state_dict(weights,strict=False)
 assert not result.unexpected_keys and all(k.startswith(('ir_backbone.','ir_encoder.','ir_samplers.','ir_corr.','ir_corr_strength')) for k in result.missing_keys),result
 print('IR_CORR_NATIVE_RGB_PARENT_LOADED',flush=True)
baseline.BaselineSolver.load_tuning_state=load_tuning
