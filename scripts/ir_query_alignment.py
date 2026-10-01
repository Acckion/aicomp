"""RGB-referenced query sampling of IR neighborhoods, using RGB GT only.

Not AR-CNN supervision: there are no IR-box labels or shift targets. Offsets
receive only ordinary RGB-coordinate detector gradients. Single-model outputs.
"""
import io
import os
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import binary_propagation
import torch
from torch import nn
import torch.nn.functional as F

import train_baseline as baseline
from src.core import register
from src.data.dataset.coco_dataset import CocoDetection
from src.zoo.dfine.dfine import DFINE

# Background workers should publish each progress line immediately. This
# applies to future imports/control jobs without changing other experiments.
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def valid_ir_mask(image, size=800):
    # Mask only near-zero components connected to the image boundary.
    # Interior dark targets remain available. This is a validity heuristic,
    # not a semantic visibility annotation or exact camera FOV calibration.
    gray=np.asarray(image.convert('L').resize((size,size),Image.Resampling.NEAREST))
    black=gray<=1
    seed=np.zeros_like(black);seed[0]=black[0];seed[-1]=black[-1]
    seed[:,0]=black[:,0];seed[:,-1]=black[:,-1]
    invalid=binary_propagation(seed,mask=black)
    return torch.from_numpy((~invalid).astype(np.float32)).unsqueeze(0)


@register()
class PairedIRCoco(CocoDetection):
    def __init__(self,img_folder,ann_file,transforms,archive,training=False,
                 size=800,return_masks=False,remap_mscoco_category=False):
        super().__init__(img_folder,ann_file,None,return_masks,remap_mscoco_category)
        assert not remap_mscoco_category
        self.training=training;self.size=size;self.archive_path=archive
        self._zip=None;self._pid=None
        with zipfile.ZipFile(archive) as source:
            self.members={Path(n).name:n for n in source.namelist() if len(Path(n).parts)>1 and Path(n).parts[-2]=='infrared'}
        assert all(Path(m['file_name']).name in self.members for m in self.coco.dataset['images'])

    def __getstate__(self):
        state=self.__dict__.copy();state['_zip']=None;state['_pid']=None;return state

    def __getitem__(self,index):
        rgb,target=super().load_item(index)
        meta=self.coco.loadImgs(self.ids[index])[0]
        if self._pid!=os.getpid():
            self._zip=zipfile.ZipFile(self.archive_path);self._pid=os.getpid()
        with Image.open(io.BytesIO(self._zip.read(self.members[Path(meta['file_name']).name]))) as image:
            ir=image.convert('RGB')
        assert rgb.size==(meta['width'],meta['height'])
        if not getattr(self,'allow_shuffled_size',False):assert ir.size==rgb.size
        mask=valid_ir_mask(ir,self.size)
        boxes=target['boxes'].as_subclass(torch.Tensor).clone()
        if self.training and random.random()<.5:
            rgb=rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            ir=ir.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.flip(-1)
            x0=boxes[:,0].clone();boxes[:,0]=rgb.width-boxes[:,2];boxes[:,2]=rgb.width-x0
        if self.training and random.random()<.5:
            rgb=ImageEnhance.Brightness(rgb).enhance(random.uniform(.8,1.2))
            rgb=ImageEnhance.Contrast(rgb).enhance(random.uniform(.8,1.2))
        def tensor(image):
            return torch.from_numpy(np.asarray(image.resize((self.size,self.size),Image.Resampling.BILINEAR)).copy()).permute(2,0,1).float()/255
        sample=torch.cat([tensor(rgb),tensor(ir),mask])
        if self.training:
            scale=boxes.new_tensor([rgb.width,rgb.height,rgb.width,rgb.height])
            boxes/=scale
            xy0,xy1=boxes[:,:2].clone(),boxes[:,2:].clone()
            boxes=torch.cat([(xy0+xy1)/2,xy1-xy0],-1)
        target['boxes']=boxes
        return sample,target


class QueryIRSampler(nn.Module):
    def __init__(self,query_dim,samples=9,max_offset=.15):
        super().__init__();self.samples=samples;self.max_offset=max_offset
        blocks=[];incoming=3
        for outgoing in [32,64,128]:
            blocks.extend([nn.Conv2d(incoming,outgoing,3,2,1,bias=False),nn.GroupNorm(8,outgoing),nn.SiLU()]);incoming=outgoing
        self.encoder=nn.Sequential(*blocks)
        self.offset=nn.Linear(query_dim,samples*2)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        self.query_projection=nn.Linear(query_dim,128,bias=False)
        self.output_projection=nn.Linear(128,query_dim,bias=False)
        nn.init.zeros_(self.output_projection.weight)
        grid=torch.tensor([(x,y)for y in [-.06,0,.06]for x in [-.06,0,.06]])
        self.register_buffer('base_offsets',grid)
        self.last_diagnostics={}

    def forward(self,queries,refs,ir,valid):
        # Use float32 grid_sample: AMP grid gradients can be unstable near
        # nearly-empty validity masks; return the query's original dtype.
        feature=self.encoder(ir)
        offsets=self.offset(queries).reshape(*queries.shape[:2],self.samples,2).float().tanh()*self.max_offset
        positions=refs.sigmoid()[...,:2].float().unsqueeze(2)+self.base_offsets+offsets
        grid=positions*2-1
        with torch.autocast(device_type=queries.device.type,enabled=False):
            sampled=F.grid_sample(feature.float(),grid,align_corners=False,padding_mode='zeros').permute(0,2,3,1)
            visibility=F.grid_sample(valid.float(),grid,align_corners=False,padding_mode='zeros')[:,0]
            keys=self.query_projection(queries.float()).unsqueeze(2)
            attention=(keys*sampled).sum(-1)/(128**.5)
            attention=attention.masked_fill(visibility<.5,-1e4).softmax(-1)
            attention=attention*(visibility>=.5)
            attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-6)
            evidence=(sampled*attention.unsqueeze(-1)).sum(2)
            residual=self.output_projection(evidence)
        if not self.training:
            self.last_diagnostics={'mean_learned_offset':float(offsets.abs().mean()),
                                   'queries_with_valid_ir':float((visibility>=.5).any(-1).float().mean())}
        return queries+residual.to(queries.dtype)


@register()
class IRAlignedDFINE(DFINE):
    def __init__(self,backbone,encoder,decoder,ir_enabled=True):
        super().__init__(backbone,encoder,decoder)
        self.ir_query=QueryIRSampler(decoder.dec_score_head[-1].in_features)
        self.ir_enabled=ir_enabled;self.ir_mode='paired';self._ir=None;self._valid=None
        # Registered bound method deep-copies with EMA model ownership.
        self.decoder.decoder.register_forward_pre_hook(self._inject_ir)

    def _inject_ir(self,module,inputs):
        if not self.ir_enabled:return inputs
        queries=self.ir_query(inputs[0],inputs[1],self._ir,self._valid)
        return (queries,*inputs[1:])

    def forward(self,x,targets=None):
        assert x.shape[1]==7,'Paired RGB3/IR3/valid1 input required'
        self._ir=x[:,3:6];self._valid=x[:,6:7]
        if self.ir_mode=='zero':self._ir=torch.zeros_like(self._ir);self._valid=torch.zeros_like(self._valid)
        try:return super().forward(x[:,:3],targets)
        finally:self._ir=None;self._valid=None


_original_load=baseline.BaselineSolver.load_tuning_state
def load_tuning_state(self,path):
    if not isinstance(self.model,IRAlignedDFINE):return _original_load(self,path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    weights=dict(checkpoint['ema']['module']if'ema'in checkpoint else checkpoint['model'])
    current=self.model.state_dict()
    for key in ('decoder.anchors','decoder.valid_mask'):
        if key in current:weights[key]=current[key]
    matched,info=self._matched_state(current,weights)
    assert not[k for k in info['missed']if not k.startswith('ir_query.')],info
    assert not info['unmatched'],info
    self.model.load_state_dict(matched,strict=False)
    print('IR_QUERY_INIT native detector loaded exactly; zero residual initialization',info,flush=True)
baseline.BaselineSolver.load_tuning_state=load_tuning_state

# The upstream first-batch montage accepts <=4 channels. Keep this adaptation
# local to the paired-input worker and visualize only the RGB reference image.
import src.solver.det_engine as _det_engine
_original_save_samples=_det_engine.save_samples
def _save_rgb_samples(samples,*args,**kwargs):
    return _original_save_samples(samples[:,:3],*args,**kwargs)
_det_engine.save_samples=_save_rgb_samples
