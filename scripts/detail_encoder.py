"""Inject stride-4 RGB detail into stride-8 features with a zero-start gate.

The three-level decoder is unchanged. This is a shallow-detail ablation, not a
four-level P2 decoder. A zero gate preserves the original pretrained function.
"""
import torch
from torch import nn
from src.core import register
from src.zoo.dfine.hybrid_encoder import HybridEncoder


@register()
class DetailHybridEncoder(HybridEncoder):
    def __init__(self, in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32],
                 hidden_dim=384, nhead=8, dim_feedforward=2048, dropout=0.0,
                 enc_act='gelu', use_encoder_idx=[2], num_encoder_layers=1,
                 pe_temperature=10000, expansion=1.0, depth_mult=1.0,
                 act='silu', eval_spatial_size=None, detail_channels=128):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead,
                         dim_feedforward, dropout, enc_act, use_encoder_idx,
                         num_encoder_layers, pe_temperature, expansion,
                         depth_mult, act, eval_spatial_size)
        self.detail_branch = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, stride=2, padding=1,
                      groups=detail_channels, bias=False),
            nn.BatchNorm2d(detail_channels), nn.SiLU(),
            nn.Conv2d(detail_channels, in_channels[0], 1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
        )
        self.detail_gate = nn.Parameter(torch.zeros(()))

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('DetailHybridEncoder requires backbone stages [0,1,2,3]')
        detail = self.detail_branch(feats[0])
        if detail.shape != feats[1].shape:
            raise ValueError(f'Detail / stride-8 mismatch: {detail.shape}, {feats[1].shape}')
        return super().forward([feats[1] + self.detail_gate.tanh() * detail, *feats[2:]])
