"""Frozen domain-IR spatial encoder and content-conditioned coordinate attention."""
from copy import deepcopy
from pathlib import Path
import os
import torch
from torch import nn
import torch.nn.functional as F
import ir_query_alignment as paired
import train_baseline as baseline
from src.core import register
from src.zoo.dfine.dfine import DFINE

class ContentSampler(nn.Module):
    def __init__(self,dim=256,feature_dim=384):
        super().__init__()
        self.key=nn.Linear(feature_dim,64,bias=False);self.query=nn.Linear(dim,64,bias=False)
        self.position=nn.Sequential(nn.Linear(6,64),nn.SiLU(),nn.Linear(64,64))
        self.value=nn.Linear(feature_dim,dim,bias=False);self.position_value=nn.Linear(6,dim,bias=False)
        self.shift=nn.Linear(64,2);nn.init.zeros_(self.shift.weight);nn.init.zeros_(self.shift.bias)
        self.output=nn.Linear(dim,dim,bias=False);nn.init.zeros_(self.output.weight)
        self.register_buffer('grid',torch.tensor([(x,y) for y in [-1,-.5,0,.5,1] for x in [-1,-.5,0,.5,1]],dtype=torch.float32))
    def sample(self,feature,valid,positions):
        grid=positions*2-1
        x=F.grid_sample(feature.float(),grid,align_corners=False).permute(0,2,3,1)
        v=F.grid_sample(valid.float(),grid,align_corners=False)[:,0]>=.5
        return x,v
    def forward(self,q,refs,feature,valid):
        with torch.autocast(device_type=q.device.type,enabled=False):
            q32=q.float();box=refs[:, :, 0].float();center=box[...,:2]
            # Full-image minimum radius covers modest camera shifts; box-scaled
            # extension keeps larger objects' local context. No IR GT assumed.
            radius=(box[...,2:]*.5).clamp(min=.025,max=.12)
            offsets=self.grid[None,None]*radius[:,:,None]
            first,v=self.sample(feature,valid,center[:,:,None]+offsets)
            query=self.query(q32)[:,:,None];keys=self.key(first)
            attn=(query*keys).sum(-1)/8
            attn=attn.masked_fill(~v,-1e4).softmax(-1)*v
            attn=attn/attn.sum(-1,keepdim=True).clamp_min(1e-6)
            # IR content and RGB query jointly condition displacement.
            joint=((keys*query)*attn[...,None]).sum(2)
            shift=self.shift(joint).tanh()*.06
            positions=center[:,:,None]+offsets+shift[:,:,None]
            x,v=self.sample(feature,valid,positions)
            relative=positions-center[:,:,None]
            geom=torch.cat([relative,box[...,2:][:,:,None].expand(-1,-1,25,-1),positions],-1)
            logits=(query*(self.key(x)+self.position(geom))).sum(-1)/8
            a=logits.masked_fill(~v,-1e4).softmax(-1)*v
            a=a/a.sum(-1,keepdim=True).clamp_min(1e-6)
            evidence=((self.value(x)+self.position_value(geom))*a[...,None]).sum(2)
            return q+self.output(evidence).to(q.dtype)

@register()
class IRContentDFINE(DFINE):
    def __init__(self,backbone,encoder,decoder,ir_enabled=True,ir_checkpoint='/home/fbohan/AIC/runs/ir_detector800/weights_epoch_024.pth'):
        super().__init__(backbone,encoder,decoder)
        self.ir_backbone=deepcopy(backbone);self.ir_encoder=deepcopy(encoder)
        source=Path(ir_checkpoint);cache=Path('/dev/shm/aicomp_ir_content_encoder.pth')
        signature={'source':str(source.resolve()),'bytes':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns}
        cached=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else None
        if cached is not None and cached.get('signature')==signature:
            weights=cached['weights']
        else:
            checkpoint=torch.load(ir_checkpoint,map_location='cpu',weights_only=False)
            full=checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
            weights={k:v for k,v in full.items() if k.startswith(('backbone.','encoder.'))}
            tmp=cache.with_suffix(f'.{os.getpid()}.tmp');torch.save({'signature':signature,'weights':weights},tmp);tmp.replace(cache)
        for prefix,branch in [('backbone.',self.ir_backbone),('encoder.',self.ir_encoder)]:
            sub={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
            branch.load_state_dict(sub,strict=True);branch.requires_grad_(False)
        self.ir_encoder.eval_spatial_size=None
        self.ir_teacher_source=ir_checkpoint
        self.ir_samplers=nn.ModuleList([ContentSampler(decoder.dec_score_head[-1].in_features,encoder.hidden_dim) for _ in range(3)])
        self.ir_enabled=ir_enabled;self.ir_mode='paired';self._feature=None;self._valid=None
        for i,layer in enumerate(self.decoder.decoder.layers[-3:]):
            layer._ir_content_index=i
            layer.register_forward_pre_hook(self._inject)
        print('IR_CONTENT_TEACHER strict backbone/encoder loaded',ir_checkpoint,flush=True)
    def _inject(self,module,inputs):
        if not self.ir_enabled:return inputs
        return (self.ir_samplers[module._ir_content_index](inputs[0],inputs[1],self._feature,self._valid),*inputs[1:])
    def forward(self,x,targets=None):
        assert x.shape[1]==7
        self._valid=x[:,6:7]
        try:
            if self.ir_enabled:
                self.ir_backbone.eval();self.ir_encoder.eval()
                ir=x[:,3:6]
                if self.ir_mode=='zero':ir=torch.zeros_like(ir);self._valid=torch.zeros_like(self._valid)
                with torch.no_grad():self._feature=self.ir_encoder(self.ir_backbone(ir))[0].detach()
            return super().forward(x[:,:3],targets)
        finally:self._feature=None;self._valid=None

_original=baseline.BaselineSolver.load_tuning_state
def load_tuning_state(self,path):
    if not isinstance(self.model,IRContentDFINE):return _original(self,path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    weights=dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model'])
    current=self.model.state_dict()
    for key in ('decoder.anchors','decoder.valid_mask'):
        if key in current:weights[key]=current[key]
    result=self.model.load_state_dict(weights,strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith(('ir_backbone.','ir_encoder.','ir_samplers.')) for k in result.missing_keys),result
    print('IR_CONTENT_INIT all native RGB parameters loaded; frozen IR preserved',flush=True)
baseline.BaselineSolver.load_tuning_state=load_tuning_state
