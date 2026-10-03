"""Prepared alternative: same-basis native detail, bounded before query selection.

Regional original/low-detail views share a frozen
current encoder; their difference isolates pixel-detail changes from crop context.
"""
import torch
from torch import nn
from torch.nn import functional as F
import native_grid
from src.core import register
from src.zoo.dfine.dfine import DFINE
import train_baseline as baseline


@register()
class NativeEncodedDeltaDFINE(DFINE):
    inference_input_protocol = 'native_rgb7_pil_bilinear_base_v1'

    def prepare_inference_input(self, image, size):
        """Match NativeRGBCoco/Collate using the original, possibly flipped PIL view."""
        import numpy as np
        from PIL import Image
        def tensor(im):
            return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
        assert image.mode == 'RGB' and size > 0
        w,h=image.size
        packed=torch.zeros(1,7,max(h,self.size),max(w,self.size))
        packed[0,:3,:self.size,:self.size]=tensor(image.resize((self.size,self.size),Image.Resampling.BILINEAR))
        packed[0,3:6,:h,:w]=tensor(image)
        packed[0,6,0,0]=h;packed[0,6,0,1]=w
        self.forced_size=size
        return packed

    def __init__(self, backbone, encoder, decoder, size=800, tile_size=800,
                 detail_enabled=True, residual_bound=.25, region_fraction=.6,
                 routing_enabled=False):
        super().__init__(backbone,encoder,decoder)
        self.size=size;self.tile_size=tile_size;self.detail_enabled=detail_enabled
        self.residual_bound=residual_bound;self.forced_size=None;self.detail_mode='real'
        assert 0 < region_fraction <= 1
        self.region_fraction=region_fraction;self.routing_enabled=routing_enabled
        self.delta_bridge=nn.Conv2d(encoder.hidden_dim,encoder.hidden_dim,1,bias=False)
        nn.init.zeros_(self.delta_bridge.weight)
        self.encoder.eval_spatial_size=None;self.decoder.eval_spatial_size=None

    def regional_delta(self,packed,shape,windows=None):
        flags=[(m,m.training) for root in [self.backbone,self.encoder] for m in root.modules()]
        self.backbone.eval();self.encoder.eval()
        maps=[]
        try:
            with torch.no_grad():
                for index,image in enumerate(packed):
                    h,w=map(int,image[6,0,:2].detach().cpu().tolist())
                    assert h>0 and w>0
                    # The global view already has at least the original pixel
                    # dimensions here. An original-minus-reconstructed delta
                    # would mostly measure interpolation rather than restore
                    # detail discarded by shrinking the original image.
                    if h <= self.size and w <= self.size:
                        maps.append(image.new_zeros((1,self.encoder.hidden_dim,*shape)))
                        continue
                    original=image[None,3:6,:h,:w]
                    low=F.interpolate(image[None,:3,:self.size,:self.size],size=(h,w),mode='bilinear',align_corners=False)
                    if self.detail_mode=='low':original=low
                    crop_h,crop_w=round(h*self.region_fraction),round(w*self.region_fraction)
                    rectangles=[(x,y,x+crop_w,y+crop_h) for y in (0,h-crop_h) for x in (0,w-crop_w)] if windows is None else [(round(a*w),round(b*h),round(c*w),round(d*h)) for a,b,c,d in windows[index]]
                    assembled=image.new_zeros((1,self.encoder.hidden_dim,*shape))
                    coverage=image.new_zeros((1,1,*shape))
                    for x,y,x2,y2 in rectangles:
                        crop_h,crop_w=y2-y,x2-x
                        def encode(source):
                            tile=F.interpolate(source[:,:,y:y+crop_h,x:x+crop_w],size=(self.tile_size,self.tile_size),mode='bilinear',align_corners=False)
                            return self.encoder(self.backbone(tile))[0]
                        actual=encode(original).float();sham=encode(low).float()
                        delta=actual-sham
                        ch,cw=shape;ya,yb=round(y/h*ch),round((y+crop_h)/h*ch);xa,xb=round(x/w*cw),round((x+crop_w)/w*cw)
                        assembled[:,:,ya:yb,xa:xb]+=F.interpolate(delta,size=(yb-ya,xb-xa),mode='bilinear',align_corners=False)
                        coverage[:,:,ya:yb,xa:xb]+=1
                    if windows is None and self.region_fraction>=.5:assert (coverage>0).all()
                    assembled=assembled/coverage.clamp_min(1)
                    if self.detail_mode=='zero':assembled=torch.zeros_like(assembled)
                    if self.detail_mode=='shuffle':
                        flat=assembled.flatten(2);assembled=flat[:,:,torch.randperm(flat.shape[-1],device=flat.device)].reshape_as(assembled)
                    maps.append(assembled)
        finally:
            for module,flag in flags:module.training=flag
        return torch.cat(maps)

    def forward(self,packed,targets=None):
        assert packed.shape[1]==7
        rgb=packed[:,:3,:self.size,:self.size]
        active=self.forced_size or self.size
        if self.training and self.forced_size is None:
            sizes=(640,704,768,800,832,896,960,992)
            active=sizes[int(torch.randint(len(sizes),()).item())]
        if active!=self.size:rgb=F.interpolate(rgb,size=(active,active),mode='bilinear',align_corners=False)
        features=list(self.encoder(self.backbone(rgb)))
        if self.detail_enabled:
            windows=None
            if self.routing_enabled:
                from native_crop_routing import encoder_candidates,select_windows
                # Routing precedes the trainable decoder. FP32 prevents a
                # no-grad AMP cast from poisoning its later gradient cache.
                with torch.no_grad(),torch.autocast(device_type=features[0].device.type,enabled=False):
                    boxes,scores=encoder_candidates(self.decoder,[v.detach().float() for v in features])
                    windows=[select_windows(b,s,self.region_fraction) for b,s in zip(boxes,scores)]
            delta=self.regional_delta(packed,features[0].shape[-2:],windows)
            residual=self.delta_bridge(delta)
            # Bound perturbation relative to each image's mature feature scale.
            scale=features[0].detach().float().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
            residual=self.residual_bound*scale*torch.tanh(residual.float()/scale)
            features[0]=features[0]+residual.to(features[0].dtype)
        return self.decoder(features,targets)


def initialize(model,path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    weights=state['ema']['module'] if 'ema' in state else state['model']
    weights=dict(weights);current=model.state_dict()
    for key in ('decoder.anchors','decoder.valid_mask'):
        if key in current:weights[key]=current[key]
    matched,info=baseline.BaselineSolver._matched_state(current,weights)
    allowed=('delta_bridge.','score_head','denoising_class_embed')
    assert all(any(k.startswith(p) or p in k for p in allowed) for k in info['missed']),info
    assert all('score_head' in k or 'denoising_class_embed' in k for k in info['unmatched']),info
    model.load_state_dict(matched,strict=False)
    assert torch.count_nonzero(model.delta_bridge.weight)==0
    print('ENCODED_DELTA_INIT',info,flush=True)


_previous=baseline.BaselineSolver.load_tuning_state
def load(solver,path):
    if isinstance(solver.model,NativeEncodedDeltaDFINE):initialize(solver.model,path)
    else:return _previous(solver,path)
baseline.BaselineSolver.load_tuning_state=load
