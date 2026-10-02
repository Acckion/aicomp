"""Object-local reliability gating after content alignment; one detector."""
import torch
from torch import nn
import torch.nn.functional as F
from ir_content_alignment import ContentSampler, IRContentDFINE
from src.core import register


class ReliabilitySampler(ContentSampler):
    def __init__(self, dim, feature_dim, enabled=True):
        super().__init__(dim, feature_dim)
        self.reliability = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        self.gate_enabled = enabled
        self.last_diagnostics = None

    def forward(self, q, refs, feature, valid):
        updated = super().forward(q, refs, feature, valid)
        if not self.gate_enabled:
            return updated
        with torch.autocast(device_type=q.device.type, enabled=False):
            box = refs[:, :, 0].float()
            radius = (box[..., 2:] * .5).clamp(.025, .12)
            positions = box[..., :2].unsqueeze(2)
            patches, mask = self.sample(feature, valid, positions + self.grid[None, None] * radius.unsqueeze(2))
            query = self.query(q.float()).unsqueeze(2)
            keys = self.key(patches)
            search = ((query * keys).sum(-1) / 8).masked_fill(~mask, -1e4).softmax(-1) * mask
            search = search / search.sum(-1, keepdim=True).clamp_min(1e-6)
            shift = self.shift(((keys * query) * search.unsqueeze(-1)).sum(2)).tanh() * .06
            aligned = positions + self.grid[None, None] * radius.unsqueeze(2) + shift.unsqueeze(2)
            patches, mask = self.sample(feature, valid, aligned)
            similarity = F.cosine_similarity(query, self.key(patches), dim=-1)
            weights = similarity.masked_fill(~mask, -1e4).softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            entropy = -(weights * weights.clamp_min(1e-6).log()).sum(-1) / 3.218875825
            support = mask.float().mean(-1)
            best = similarity.masked_fill(~mask, -1).amax(-1)
            # This measures local feature variation, not calibrated IR quality.
            mean = (patches * weights.unsqueeze(-1)).sum(2)
            variance = ((patches - mean.unsqueeze(2)).square() * weights.unsqueeze(-1)).sum(2).mean(-1)
            contrast = variance.clamp_min(1e-8).sqrt() / (mean.abs().mean(-1) + 1e-6)
            evidence = torch.stack([best, 1 - entropy, support, contrast.clamp(max=5) / 5], -1)
            gate = self.reliability(evidence).sigmoid() * (support > 0).unsqueeze(-1)
            self.last_diagnostics = {'gate_mean': gate.detach().mean(), 'support_mean': support.detach().mean()}
        return q + gate.to(q.dtype) * (updated - q)


@register()
class ReliabilityIRDFINE(IRContentDFINE):
    def __init__(self, backbone, encoder, decoder, reliability_enabled=True, modality_dropout=.15):
        super().__init__(backbone, encoder, decoder)
        self.ir_samplers = nn.ModuleList([ReliabilitySampler(decoder.dec_score_head[-1].in_features, encoder.hidden_dim, reliability_enabled) for _ in range(3)])
        self.modality_dropout = modality_dropout

    def forward(self, x, targets=None):
        if self.training and self.modality_dropout:
            x = x.clone()
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.modality_dropout).to(x.dtype)
            x[:, 3:7] *= keep
        return super().forward(x, targets)
