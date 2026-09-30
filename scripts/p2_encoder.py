"""Preserve the pretrained three-level encoder and add a real stride-4 output."""
import torch
from torch import nn
import torch.nn.functional as F
from src.core import register
from src.zoo.dfine.hybrid_encoder import HybridEncoder


@register()
class P2HybridEncoder(HybridEncoder):
    def __init__(self,in_channels=[512,1024,2048],feat_strides=[8,16,32],
                 hidden_dim=384,nhead=8,dim_feedforward=2048,dropout=0.0,
                 enc_act='gelu',use_encoder_idx=[2],num_encoder_layers=1,
                 pe_temperature=10000,expansion=1.0,depth_mult=1.0,
                 act='silu',eval_spatial_size=None,detail_channels=128,
                 freeze_bn_stats=True):
        super().__init__(in_channels,feat_strides,hidden_dim,nhead,dim_feedforward,
                         dropout,enc_act,use_encoder_idx,num_encoder_layers,
                         pe_temperature,expansion,depth_mult,act,eval_spatial_size)
        self.p2_detail=nn.Sequential(
            nn.Conv2d(detail_channels,hidden_dim,1,bias=False),
            nn.GroupNorm(32,hidden_dim),nn.SiLU(),
            nn.Conv2d(hidden_dim,hidden_dim,3,padding=1,groups=hidden_dim,bias=False),
        )
        self.p2_gate=nn.Parameter(torch.zeros(()))
        self.freeze_bn_stats=freeze_bn_stats
        self.out_channels=[hidden_dim]*4;self.out_strides=[4,8,16,32]

    def train(self,mode=True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for m in self.modules():
                if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self

    def forward(self,feats):
        if len(feats)!=4:raise ValueError('P2 requires backbone stages [0,1,2,3]')
        original=super().forward(feats[1:])
        p2=F.interpolate(original[0],size=feats[0].shape[-2:],mode='nearest')
        p2=p2+self.p2_gate.tanh()*self.p2_detail(feats[0])
        return [p2,*original]
