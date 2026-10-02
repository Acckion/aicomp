"""Use aircraft shape priors to allocate queries, without relabeling aircraft.

Official train-only source vocabulary audit found Airplane/Helicopter support
for 7 of 26 sampled UAV GT boxes. This motivates a coverage experiment, not
an equivalence between these source classes and the target UAV class.
The classifier groups, DN embeddings and total query budget stay unchanged.
"""
import torch
import torch.nn.functional as F
import taxonomy_pooling
from src.zoo.dfine.dfine_decoder import DFINETransformer

SOURCE_ROWS = (115, 264)
_original_select = DFINETransformer._select_topk


def select_air_queries(self, memory, outputs_logits, outputs_anchors_unact, topk):
    if self.query_select_method != 'default':
        return _original_select(self, memory, outputs_logits, outputs_anchors_unact, topk)
    source = self.enc_score_head.source
    assert source.weight.shape == (366, memory.shape[-1])
    budget = min(50, topk // 6)
    if budget == 0:
        return _original_select(self, memory, outputs_logits, outputs_anchors_unact, topk)
    primary = outputs_logits.amax(-1).topk(topk - budget, dim=-1).indices
    with torch.no_grad():
        scores = F.linear(memory.detach(), source.weight[list(SOURCE_ROWS)].detach(),
                          source.bias[list(SOURCE_ROWS)].detach()).amax(-1)
        # Invalid deterministic anchors must not receive the auxiliary budget.
        scores = scores.masked_fill(~outputs_anchors_unact.isfinite().all(-1), -torch.inf)
        scores.scatter_(1, primary, -torch.inf)
        values, auxiliary = scores.topk(budget, dim=-1)
        assert torch.isfinite(values).all(), 'Not enough valid auxiliary anchors'
    indices = torch.cat((primary, auxiliary), dim=-1)
    anchors = outputs_anchors_unact.gather(1, indices[..., None].expand(-1, -1, 4))
    logits = outputs_logits.gather(1, indices[..., None].expand(-1, -1, outputs_logits.shape[-1])) if self.training else None
    features = memory.gather(1, indices[..., None].expand(-1, -1, memory.shape[-1]))
    return features, logits, anchors


assert 10 not in taxonomy_pooling.GROUPS
DFINETransformer._select_topk = select_air_queries
