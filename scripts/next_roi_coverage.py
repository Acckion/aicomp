"""Same 64 crops, reserving budget for small and spatially diverse queries."""
import torch
from native_roi import NativeROIResidual, NativeROIDFINE
from src.core import register


def coverage_selection(scores, boxes, count=64):
    """No GT, class IDs or image labels participate in selection."""
    probability = scores.detach().sigmoid().amax(-1)
    n = min(count, len(probability))
    chosen = probability.topk(min(32, n)).indices.tolist()
    selected = set(chosen)
    areas = boxes.detach()[:, 2:].prod(-1)
    small = ((areas < .01) & (probability >= .03)).nonzero().flatten()
    if len(small):
        ranked = small[probability[small].argsort(descending=True)]
        added = 0
        for index in ranked.tolist():
            if index not in selected and len(chosen) < n and added < 16:
                chosen.append(index); selected.add(index); added += 1
    while len(chosen) < n:
        distance = torch.cdist(boxes[:, :2].detach().float(), boxes[chosen, :2].detach().float()).amin(-1)
        # Low scores cannot monopolize diversity; remaining budget is spatial.
        utility = probability * distance.clamp(max=.5)
        utility[chosen] = -1
        index = int(utility.argmax())
        chosen.append(index); selected.add(index)
    return torch.tensor(chosen, device=scores.device)


class CoverageResidual(NativeROIResidual):
    def forward(self, queries, refs, scores, native, sizes, num_queries):
        prefix = queries.shape[1] - num_queries
        selection_scores = torch.full_like(scores, -100.)
        for b in range(len(queries)):
            ids = coverage_selection(scores[b, prefix:], refs[b, prefix:, 0], self.topk) + prefix
            selection_scores[b, ids] = 10.
        # Parent consumes scores only for top-k crop selection, never detection.
        return super().forward(queries, refs, selection_scores, native, sizes, num_queries)


@register()
class CoverageROIDFINE(NativeROIDFINE):
    def __init__(self, backbone, encoder, decoder, coverage_enabled=True, size=800):
        super().__init__(backbone, encoder, decoder, size=size)
        if coverage_enabled:
            self.native_roi = CoverageResidual(decoder.dec_score_head[decoder.eval_idx].in_features)
