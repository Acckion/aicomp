"""One RGB detector with proposal-conditioned crops from its original image.

Training and evaluation select proposals exclusively from detector queries;
GT boxes never choose crops. DN queries are explicitly excluded. The original
classification/FDR heads supervise a zero-initialized query residual.
"""
import functools
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F
import train_baseline as baseline
from src.core import register
from src.data.dataset.coco_dataset import CocoDetection
from src.data.dataloader import BaseCollateFunction
from src.zoo.dfine.dfine import DFINE

@register()
class NativeRGBCoco(CocoDetection):
    def __init__(self,img_folder,ann_file,transforms=None,training=False,size=800,return_masks=False,remap_mscoco_category=False):
        super().__init__(img_folder,ann_file,None,return_masks,remap_mscoco_category)
        assert not remap_mscoco_category
        self.training=training;self.size=size
    def __getitem__(self,index):
        image,target=super().load_item(index)
        boxes=target['boxes'].as_subclass(torch.Tensor).clone()
        if self.training and random.random()<.5:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x0=boxes[:,0].clone();boxes[:,0]=image.width-boxes[:,2];boxes[:,2]=image.width-x0
        if self.training and random.random()<.5:
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.8,1.2))
        def tensor(im):return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
        native=tensor(image);global_rgb=tensor(image.resize((self.size,self.size),Image.Resampling.BILINEAR))
        if self.training:
            boxes/=boxes.new_tensor([image.width,image.height,image.width,image.height])
            a,b=boxes[:,:2].clone(),boxes[:,2:].clone();boxes=torch.cat([(a+b)/2,b-a],-1)
        target['boxes']=boxes
        return (global_rgb,native),target

@register()
class NativeRGBCollate(BaseCollateFunction):
    def __init__(self,base_size=800,stop_epoch=8,ema_restart_decay=.9999,base_size_repeat=None):
        super().__init__();self.base_size=base_size;self.scales=None;self.stop_epoch=stop_epoch;self.ema_restart_decay=ema_restart_decay
    def __call__(self,items):
        height=max(self.base_size,max(v[0][1].shape[-2] for v in items));width=max(self.base_size,max(v[0][1].shape[-1] for v in items))
        packed=torch.zeros(len(items),7,height,width)
        for i,((rgb,native),_) in enumerate(items):
            h,w=native.shape[-2:];packed[i,:3,:self.base_size,:self.base_size]=rgb
            packed[i,3:6,:h,:w]=native;packed[i,6,0,0]=h;packed[i,6,0,1]=w
        return packed,[v[1] for v in items]

class NativeROIResidual(nn.Module):
    def __init__(self,hidden=256,topk=64,patch=96,chunk=16):
        super().__init__();self.topk=topk;self.patch=patch;self.chunk=chunk
        self.encoder=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,128,3,2,1),nn.GroupNorm(8,128),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.spatial=nn.Linear(128*4*4,128);self.geometry=nn.Linear(4,32)
        self.fusion=nn.Sequential(nn.Linear(hidden+128+32,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)
        axis=(torch.arange(patch)+.5)/patch-.5;yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('unit_grid',torch.stack([xx,yy],-1))
    def forward(self,queries,refs,scores,native,sizes,num_queries):
        residual=torch.zeros_like(queries);prefix=queries.shape[1]-num_queries
        assert prefix>=0
        selected=scores[:,prefix:].detach().sigmoid().amax(-1).topk(min(self.topk,num_queries),dim=1).indices+prefix
        for b in range(len(queries)):
            for ids in selected[b].split(self.chunk):
                boxes=refs[b,ids,0].detach().float();h,w=sizes[b];ph,pw=native.shape[-2:]
                # Twice the predicted extent, with an 8-native-pixel minimum.
                span=torch.maximum(boxes[:,2:]*2,boxes.new_tensor([8/w,8/h]))
                coordinates=boxes[:,:2,None,None].permute(0,2,3,1)+self.unit_grid[None]*span[:,None,None]
                coordinates=coordinates*coordinates.new_tensor([w/pw,h/ph]);grid=(coordinates*2-1).reshape(1,-1,self.patch,2)
                with torch.autocast(device_type=queries.device.type,enabled=False):
                    patches=F.grid_sample(native[b:b+1].float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
                    patches=patches.reshape(3,len(ids),self.patch,self.patch).permute(1,0,2,3)
                feature=self.spatial(self.encoder(patches).flatten(1));geometry=self.geometry(boxes)
                residual[b,ids]=self.fusion(torch.cat([queries[b,ids],feature,geometry],-1)).to(queries.dtype)
        return queries+residual

@register()
class NativeROIDFINE(DFINE):
    def __init__(self,backbone,encoder,decoder,roi_enabled=True,size=800,roi_topk=64):
        super().__init__(backbone,encoder,decoder);self.roi_enabled=roi_enabled;self.size=size
        self.native_roi=NativeROIResidual(decoder.dec_score_head[decoder.eval_idx].in_features,roi_topk)
        self._native=None;self._sizes=None
        self.decoder.decoder.layers[self.decoder.eval_idx].register_forward_hook(self._inject_roi)
    def _inject_roi(self,module,args,queries):
        if not self.roi_enabled:return queries
        scores=self.decoder.dec_score_head[self.decoder.eval_idx](queries)
        return self.native_roi(queries,args[1],scores,self._native,self._sizes,self.decoder.num_queries)
    def forward(self,x,targets=None):
        assert x.shape[1]==7
        self._native=x[:,3:6];self._sizes=[(int(v[0]),int(v[1])) for v in x[:,6,0,:2].detach().cpu()]
        try:return super().forward(x[:,:3,:self.size,:self.size],targets)
        finally:self._native=None;self._sizes=None

_original_load=baseline.BaselineSolver.load_tuning_state
def load(solver,path):
    if not isinstance(solver.model,NativeROIDFINE):return _original_load(solver,path)
    state=torch.load(path,map_location='cpu',weights_only=False);weights=dict(state['ema']['module'] if 'ema' in state else state['model'])
    current=solver.model.state_dict()
    for key in ('decoder.anchors','decoder.valid_mask'):weights[key]=current[key]
    missing=set(current)-set(weights);assert all(k.startswith('native_roi.') for k in missing),missing
    assert not set(weights)-set(current)
    result=solver.model.load_state_dict(weights,strict=False);assert set(result.missing_keys)==missing and not result.unexpected_keys
    print(f'NATIVE_ROI_INIT {len(weights)} original tensors exact; zero residual',flush=True)
baseline.BaselineSolver.load_tuning_state=load
import src.solver.det_engine as det_engine
_original_save=det_engine.save_samples
def save(samples,*args,**kwargs):return _original_save(samples[:,:3,:800,:800],*args,**kwargs)
det_engine.save_samples=save
