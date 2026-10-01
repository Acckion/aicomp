"""Scale-normalized auxiliary regression for a controlled D-FINE ablation.

This does not change the matcher, GO-LSD indices, classification, FGL or model.
For matched normalized cxcywh boxes p and t, let d=max(t.wh, 16/800).
The added loss is .25 * sum_i mean_4 SmoothL1((p_i-t_i)/[dx,dy,dx,dy], beta=.1)
/ N_go. Denominator flooring and SmoothL1's bounded derivative prevent tiny
boxes from producing unbounded gradients. Each original regression branch
(final, auxiliary, pre, encoder and denoising) receives this same auxiliary.
"""
import torch
import torch.nn.functional as F

from src.core import register
from src.zoo.dfine.dfine_criterion import DFINECriterion


def relative_box_loss(pred_boxes, target_boxes, num_boxes, min_extent=16 / 800,
                      beta=0.1):
    """Unweighted auxiliary; reductions are FP32 even under model AMP."""
    if min_extent <= 0 or beta <= 0:
        raise ValueError('min_extent and beta must be positive')
    if pred_boxes.numel() == 0:
        return pred_boxes.float().sum() * 0.0
    pred_boxes = pred_boxes.float()
    target_boxes = target_boxes.detach().float()
    scale = target_boxes[:, 2:].clamp_min(min_extent).repeat(1, 2)
    residual = (pred_boxes - target_boxes) / scale
    losses = F.smooth_l1_loss(residual, torch.zeros_like(residual),
                              reduction='none', beta=beta)
    return losses.mean(dim=-1).sum() / max(float(num_boxes), 1.0)


@register()
class RelativeBoxCriterion(DFINECriterion):
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0,
                 num_classes=80, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, relative_weight=0.25,
                 relative_min_extent=16 / 800, relative_beta=0.1):
        if relative_weight < 0 or relative_min_extent <= 0 or relative_beta <= 0:
            raise ValueError('Invalid relative regression settings')
        weight_dict = dict(weight_dict)
        weight_dict['loss_relative_box'] = relative_weight
        super().__init__(matcher, weight_dict, losses, alpha, gamma,
                         num_classes, reg_max, boxes_weight_format,
                         share_matched_indices)
        self.relative_min_extent = relative_min_extent
        self.relative_beta = relative_beta

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        losses = super().loss_boxes(outputs, targets, indices, num_boxes,
                                    boxes_weight=boxes_weight)
        idx = self._get_src_permutation_idx(indices)
        src = outputs['pred_boxes'][idx]
        target = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)])
        # DFINECriterion.forward applies loss_relative_box's own .25 weight.
        # It never inherits the existing loss_bbox weight of 5.
        losses['loss_relative_box'] = relative_box_loss(
            src, target, num_boxes, self.relative_min_extent, self.relative_beta)
        return losses
