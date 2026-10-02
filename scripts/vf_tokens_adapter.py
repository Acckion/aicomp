"""Frozen current-image local CLIP patch evidence; one RGB detector output.

Four regional crops retain spatial patch positions. Frozen public teacher is
process-shared outside the trainable state/EMA; the checkpoint names the teacher.
No GT or test-time optimization selects its features.
"""
import torch
from torch import nn
from torch.nn import functional as F
import train_baseline as baseline
from src.core import register
from src.zoo.dfine.dfine import DFINE
_TEACHERS={}
TEACHER='/home/fbohan/AIC/checkpoints/clip/clip-vit-large-patch14-336'
def teacher(device):
    key=str(device)
    if key not in _TEACHERS:
        from transformers import CLIPVisionModel
        # Loading a frozen branch must not consume the control's data/aug RNG.
        with torch.random.fork_rng(devices=[device.index or 0]):
            t=CLIPVisionModel.from_pretrained(TEACHER,local_files_only=True,torch_dtype=torch.float16).to(device).eval().requires_grad_(False)
        _TEACHERS[key]=t
    return _TEACHERS[key]

@torch.no_grad()
def patch_map(x):
    t=teacher(x.device);out=[]
    mean=x.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
    std=x.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    h,w=x.shape[-2:]
    for image in x:
        tiles=[]
        for row in range(2):
            for col in range(2):
                crop=image[None,:,row*h//2:(row+1)*h//2,col*w//2:(col+1)*w//2]
                crop=(F.interpolate(crop,size=(336,336),mode='bicubic',align_corners=False).clamp(0,1)-mean)/std
                with torch.autocast('cuda',dtype=torch.float16):tokens=t(crop).last_hidden_state[:,1:]
                tiles.append(tokens.transpose(1,2).reshape(1,1024,24,24))
        out.append(torch.cat([torch.cat(tiles[:2],3),torch.cat(tiles[2:],3)],2))
    return torch.cat(out,0)

class PatchResidual(nn.Module):
    def __init__(self,channels):
        super().__init__();self.project=nn.Conv2d(1024,96,1,bias=False);self.norm=nn.GroupNorm(12,96,affine=False);self.local=nn.Conv2d(96,96,3,padding=1,groups=96,bias=False);self.output=nn.Conv2d(96,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,x,patch):
        with torch.autocast('cuda',enabled=False):
            p=self.project(patch.float());p=F.gelu(self.norm(p));p=self.local(p)
            p=F.interpolate(p,size=x.shape[-2:],mode='bilinear',align_corners=False)
            y=self.output(p)
        return x+y.to(x.dtype)

@register()
class VFTokensDFINE(DFINE):
    def __init__(self,backbone,encoder,decoder,vf_enabled=True):
        super().__init__(backbone,encoder,decoder);self.vf_enabled=vf_enabled;self.vf_mode='real';self.vf_adapter=PatchResidual(encoder.hidden_dim);self._vf_patch=None
        self.encoder.input_proj[0].register_forward_hook(self._inject)
    def _inject(self,module,args,output):
        if self._vf_patch is None:return output
        return self.vf_adapter(output,self._vf_patch)
    def forward(self,x,targets=None):
        try:
            if self.vf_enabled:
                self._vf_patch=patch_map(x)
                if self.vf_mode=='zero':self._vf_patch=torch.zeros_like(self._vf_patch)
                elif self.vf_mode=='shuffle':
                    shape=self._vf_patch.shape;v=self._vf_patch.flatten(2);self._vf_patch=v[:,:,torch.randperm(v.shape[-1],device=v.device)].reshape(shape)
                elif self.vf_mode=='wrong_image':
                    assert x.shape[0]>=2, 'wrong-image roll requires >=2 distinct images'
                    self._vf_patch=self._vf_patch.roll(1,0)
            return super().forward(x,targets)
        finally:self._vf_patch=None

_original=baseline.BaselineSolver.load_tuning_state
def load(solver,path):
    if not isinstance(solver.model,VFTokensDFINE):return _original(solver,path)
    state=torch.load(path,map_location='cpu',weights_only=False);weights=dict(state['ema']['module'] if 'ema' in state else state['model']);current=solver.model.state_dict()
    for key in ('decoder.anchors','decoder.valid_mask'):weights[key]=current[key]
    missing=set(current)-set(weights);assert all(k.startswith('vf_adapter.') for k in missing),missing
    assert not set(weights)-set(current)
    result=solver.model.load_state_dict(weights,strict=False);assert set(result.missing_keys)==missing and not result.unexpected_keys
    print('VF_INIT original RGB tensors matched; frozen current-image spatial residual zero initialized',flush=True)
baseline.BaselineSolver.load_tuning_state=load
