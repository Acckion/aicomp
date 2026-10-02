"""Adapt the IR neck through final RGB-coordinate detection losses only.

Same parameters and checkpoint layout as IRContentDFINE. The IR backbone stays
frozen; BN statistics stay fixed. The control changes only neck trainability.
"""
import torch
import os
from pathlib import Path
from ir_content_alignment import IRContentDFINE
import train_baseline as baseline
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


_load_tuning = baseline.BaselineSolver.load_tuning_state


def load_joint_parent(self, path):
    if not isinstance(self.model, IRJointDFINE):
        return _load_tuning(self, path)
    # Retain the original checkpoint remotely; cache only its unmodified EMA
    # weights so successive smoke/train/control starts do not reread optimizer
    # state through SSHFS. A source signature prevents reuse after replacement.
    source = Path(path)
    signature = {'source': str(source.resolve()), 'bytes': source.stat().st_size,
                 'mtime_ns': source.stat().st_mtime_ns}
    cache = Path('/dev/shm/aicomp_ir_joint_parent.pth')
    existing = torch.load(cache, map_location='cpu', weights_only=False) if cache.exists() else None
    if existing is None or existing.get('source_signature') != signature:
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        weights = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
        torch.save({'model': weights, 'source_signature': signature}, temporary)
        temporary.replace(cache)
        del checkpoint, weights
    del existing
    return _load_tuning(self, cache)


baseline.BaselineSolver.load_tuning_state = load_joint_parent
