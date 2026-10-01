"""DEIM Matchability-Aware Loss with the mature D-FINE localization path.

Formula copied from DEIM authors' DEIMv2/engine/deim/deim_criterion.py
loss_labels_mal (source revision 1d2ca42171570c713e78fc6a766ec5104b7f4724).
Positive BCE target = detached IoU ** gamma, positive weight = 1;
negative target = 0, weight = detached sigmoid(logit) ** gamma.
Only classification changes. Returning loss_vfl preserves D-FINE's auxiliary,
encoder, pre-decoder and denoising dispatch/weighting and all FGL/GO-LSD losses.
This is MAL alone, not a reproduction of the full DEIM training recipe.
"""
import torch
import torch.nn.functional as F
from src.core import register
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou


@register()
class MALDFINECriterion(DFINECriterion):
    def __init__(self, matcher, weight_dict, losses, alpha=0.75, gamma=2.0,
                 num_classes=12, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, mal_alpha=None):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes,
                         reg_max, boxes_weight_format, share_matched_indices)
        self.mal_alpha = mal_alpha

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            predicted = outputs['pred_boxes'][idx]
            truth = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)])
            ious = torch.diag(box_iou(box_cxcywh_to_xyxy(predicted),
                                     box_cxcywh_to_xyxy(truth))[0]).detach()
        else:
            ious = values
        logits = outputs['pred_logits']
        classes = torch.full(logits.shape[:2], self.num_classes,
                             dtype=torch.long, device=logits.device)
        classes[idx] = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        positive = F.one_hot(classes, self.num_classes + 1)[..., :-1]
        scores = torch.zeros_like(classes, dtype=logits.dtype)
        scores[idx] = ious.to(scores.dtype)
        soft_target = (scores.unsqueeze(-1) * positive).pow(self.gamma)
        negative_weight = logits.sigmoid().detach().pow(self.gamma)
        if self.mal_alpha is not None:
            negative_weight = negative_weight * self.mal_alpha
        weight = negative_weight * (1 - positive) + positive
        loss = F.binary_cross_entropy_with_logits(logits, soft_target, weight=weight,
                                                  reduction='none')
        return {'loss_vfl': loss.mean(1).sum() * logits.shape[1] / num_boxes}
