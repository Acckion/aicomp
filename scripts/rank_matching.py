"""Training-only high-order matching and a gated unique-query ranking ablation.

Matching uses -p(class)*IoU**4, inspired by Rank-DETR (arXiv:2310.08854).
This is a component ablation, not a full Rank-DETR implementation. Original
D-FINE VFL/FGL/GIoU/GO-LSD and inference architecture remain unchanged.
The optional pair loss is our own hypothesis: prefer the matched, more accurate
query over unmatched competitors belonging geometrically to the same GT.
Queries assigned to any other GT are protected, even for overlapping objects.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.core import register
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou
from src.zoo.dfine.dfine_criterion import DFINECriterion


@register()
class HighOrderMatcher(HungarianMatcher):
    __share__ = ['use_focal_loss']

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25,
                 gamma=2.0, iou_power=4.0):
        super().__init__(weight_dict, use_focal_loss, alpha, gamma)
        if iou_power <= 0:
            raise ValueError('iou_power must be positive')
        self.iou_power = iou_power

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        result = []
        for logits, boxes, target in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
            labels = target['labels']
            if not len(labels):
                empty = torch.empty(0, dtype=torch.int64)
                result.append((empty, empty.clone()))
                continue
            probs = logits.float().sigmoid() if self.use_focal_loss else logits.float().softmax(-1)
            iou = box_iou(box_cxcywh_to_xyxy(boxes.detach().float()),
                          box_cxcywh_to_xyxy(target['boxes'].float()))[0].clamp(0, 1)
            cost = -(probs[:, labels] * iou.pow(self.iou_power))
            if not torch.isfinite(cost).all():
                raise RuntimeError('Non-finite high-order matching cost')
            cost = cost.cpu().numpy()
            rounds = int(return_topk) if return_topk else 1
            query_indices, gt_indices = [], []
            for _ in range(rounds):
                available = cost.min(axis=1) < 1e6
                if not available.any():
                    break
                available_indices = available.nonzero()[0]
                q, g = linear_sum_assignment(cost[available])
                q = available_indices[q]
                query_indices.extend(q.tolist()); gt_indices.extend(g.tolist())
                cost[q, :] = 1e6
            result.append((torch.tensor(query_indices, dtype=torch.int64),
                           torch.tensor(gt_indices, dtype=torch.int64)))
        return {'indices_o2m' if return_topk else 'indices': result}


def unique_query_rank_loss(outputs, targets, indices, margin=0.2,
                           quality_gap=0.05, minimum_iou=0.3,
                           minimum_winner_iou=0.5, max_competitors=5):
    logits = outputs['pred_logits'].float()
    loss = logits.sum() * 0.0
    groups = 0
    pairs = 0
    for b, (query_indices, gt_indices) in enumerate(indices):
        if not len(gt_indices):
            continue
        with torch.no_grad():
            iou = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][b].detach().float()),
                          box_cxcywh_to_xyxy(targets[b]['boxes'].float()))[0].clamp(0, 1)
            nearest = iou.argmax(dim=1)
            matched = torch.zeros(len(iou), dtype=torch.bool, device=iou.device)
            matched[query_indices.to(iou.device)] = True
        for q, g in zip(query_indices.tolist(), gt_indices.tolist()):
            with torch.no_grad():
                candidates = (~matched & (nearest == g) & (iou[:, g] >= minimum_iou)
                              & (iou[q, g] >= minimum_winner_iou)
                              & (iou[q, g] >= iou[:, g] + quality_gap)).nonzero().flatten()
            if not len(candidates):
                continue
            label = int(targets[b]['labels'][g])
            # Hard competitors are selected without gradients; both selected
            # scores receive gradients. Geometry only gates eligible pairs.
            chosen = candidates[logits[b, candidates, label].detach().topk(
                min(max_competitors, len(candidates))).indices]
            loss = loss + F.softplus(logits[b, chosen, label] - logits[b, q, label] + margin).mean()
            groups += 1
            pairs += len(chosen)
    return loss / max(groups, 1), pairs


@register()
class QueryRankCriterion(DFINECriterion):
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0,
                 num_classes=80, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, rank_weight=0.1):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes,
                         reg_max, boxes_weight_format, share_matched_indices)
        if rank_weight < 0:
            raise ValueError('rank_weight must be nonnegative')
        self.rank_weight = rank_weight
        self.last_rank_pairs = 0

    def forward(self, outputs, targets, **kwargs):
        losses = super().forward(outputs, targets, **kwargs)
        final = {k: outputs[k] for k in ('pred_logits', 'pred_boxes')}
        indices = self.matcher(final, targets)['indices']
        loss, self.last_rank_pairs = unique_query_rank_loss(final, targets, indices)
        losses['loss_query_rank'] = self.rank_weight * loss
        return losses
