"""Adapt the IR neck through final RGB-coordinate detection losses only.

Same parameters and checkpoint layout as IRContentDFINE. The IR backbone stays
frozen; BN statistics stay fixed. The control changes only neck trainability.
"""
import torch
from ir_content_alignment import IRContentDFINE
from src.core import register
from src.zoo.dfine.dfine import DFINE


@register()
class IRJointDFINE(IRContentDFINE):
    def __init__(self, backbone, encoder, decoder, ir_enabled=True,
                 adapt_ir_neck=True,
                 ir_checkpoint='/home/fbohan/AIC/runs/ir_detector800/weights_epoch_024.pth'):
        super().__init__(backbone, encoder, decoder, ir_enabled, ir_checkpoint)
        self.adapt_ir_neck = adapt_ir_neck
        self.ir_encoder.requires_grad_(adapt_ir_neck)

    def forward(self, x, targets=None):
        assert x.shape[1] == 7
        self._valid = x[:, 6:7]
        try:
            if self.ir_enabled:
                # eval fixes normalization/dropout, without disabling gradients.
                self.ir_backbone.eval(); self.ir_encoder.eval()
                ir = x[:, 3:6]
                if self.ir_mode == 'zero':
                    ir = torch.zeros_like(ir)
                    self._valid = torch.zeros_like(self._valid)
                with torch.no_grad():
                    features = self.ir_backbone(ir)
                if self.adapt_ir_neck:
                    self._feature = self.ir_encoder(features)[0]
                else:
                    with torch.no_grad():
                        self._feature = self.ir_encoder(features)[0].detach()
            return DFINE.forward(self, x[:, :3], targets)
        finally:
            self._feature = None; self._valid = None
