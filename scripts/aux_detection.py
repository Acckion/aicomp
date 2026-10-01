"""Training-only dense encoder supervision, inspired by Co-DETR's auxiliary heads.

This implements one ATSS-derived head, not Co-DETR's positive-query generation.
Priors use four stride units; candidate top-nine per level and mean+std IoU
threshold follow ATSS. Tiny GTs with zero positives receive their two nearest
P3 locations. After overlap conflicts, free nearest P3 locations restore at
least two positives per GT when sufficient locations exist. These are explicit
departures from ATSS to retain sub-stride objects and overlapping targets.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from src.core import register
from src.zoo.dfine.dfine import DFINE
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


class DenseAuxHead(nn.Module):
    def __init__(self, channels=384, num_classes=12, topk=9):
        super().__init__()
        self.num_classes, self.topk = num_classes, topk
        self.tower = nn.Sequential(nn.Conv2d(channels, 128, 3, padding=1),
                                   nn.GroupNorm(16, 128), nn.SiLU(),
                                   nn.Conv2d(128, 128, 3, padding=1),
                                   nn.GroupNorm(16, 128), nn.SiLU())
        self.cls = nn.Conv2d(128, num_classes, 1)
        self.reg = nn.Conv2d(128, 4, 1)
        nn.init.constant_(self.cls.bias, -math.log(99))
        nn.init.zeros_(self.reg.weight)
        nn.init.zeros_(self.reg.bias)
        self.last_statistics = {}

    def predictions(self, features):
        logits, boxes, priors, ranges = [], [], [], []
        offset = 0
        for feat in features:
            height, width = feat.shape[-2:]
            ys, xs = torch.meshgrid(torch.arange(height, device=feat.device),
                                    torch.arange(width, device=feat.device), indexing='ij')
            centres = torch.stack(((xs + .5) / width, (ys + .5) / height), -1).flatten(0, 1)
            sizes = torch.tensor([4 / width, 4 / height], device=feat.device).expand_as(centres)
            prior = torch.cat((centres, sizes), -1)
            hidden = self.tower(feat)
            logits.append(self.cls(hidden).flatten(2).transpose(1, 2))
            delta = self.reg(hidden).flatten(2).transpose(1, 2).float()
            xy = centres + delta[..., :2] * sizes / 4
            wh = sizes * delta[..., 2:].clamp(-6, 6).exp()
            boxes.append(torch.cat((xy, wh), -1))
            priors.append(prior)
            ranges.append((offset, offset + height * width))
            offset += height * width
        return torch.cat(logits, 1).float(), torch.cat(boxes, 1), torch.cat(priors), ranges

    @torch.no_grad()
    def assign(self, priors, ranges, gt):
        count = len(priors)
        assigned = torch.full((count,), -1, dtype=torch.long, device=priors.device)
        if not len(gt):
            return assigned, 0
        ious = box_iou(box_cxcywh_to_xyxy(priors), box_cxcywh_to_xyxy(gt))[0]
        distance = ((priors[:, None, :2] - gt[None, :, :2]) ** 2).sum(-1)
        candidates = torch.cat([distance[a:b].topk(min(self.topk, b-a), dim=0, largest=False).indices + a
                                for a, b in ranges], 0)
        gt_index = torch.arange(len(gt), device=gt.device)[None].expand_as(candidates)
        candidate_iou = ious[candidates, gt_index]
        threshold = candidate_iou.mean(0) + candidate_iou.std(0, unbiased=False)
        xyxy = box_cxcywh_to_xyxy(gt)
        inside = ((priors[:, None, :2] >= xyxy[None, :, :2]) &
                  (priors[:, None, :2] <= xyxy[None, :, 2:])).all(-1)
        positive = torch.zeros_like(inside)
        positive[candidates, gt_index] = candidate_iou >= threshold[None]
        positive &= inside
        missing = ~positive.any(0)
        fallback = int(missing.sum())
        if missing.any():
            a, b = ranges[0]
            nearest = distance[a:b, missing].topk(min(2, b-a), dim=0, largest=False).indices + a
            fallback_gt = torch.arange(len(gt), device=gt.device)[missing][None].expand_as(nearest)
            positive[nearest, fallback_gt] = True
        # A location has one owner; ties use prior IoU, never duplicate targets.
        value, owner = ious.masked_fill(~positive, -1).max(1)
        assigned[value >= 0] = owner[value >= 0]
        counts = torch.bincount(assigned[assigned >= 0], minlength=len(gt))
        needs_rescue = torch.nonzero(counts < 2).flatten()
        a, b = ranges[0]
        for index in needs_rescue:
            available = assigned[a:b] < 0
            number = min(2 - int(counts[index]), int(available.sum()))
            if not number: continue
            nearest = distance[a:b, index].masked_fill(~available, float('inf')).topk(number, largest=False).indices + a
            assigned[nearest] = index
        fallback = int((missing | (counts < 2)).sum())
        return assigned, fallback

    def forward(self, features, targets):
        logits, boxes, priors, ranges = self.predictions(features)
        classification = torch.zeros_like(logits)
        positive_boxes, target_boxes, per_gt = [], [], []
        fallbacks, gt_count = 0, 0
        for batch, target in enumerate(targets):
            gt = target['boxes'].float()
            owner, fallback = self.assign(priors, ranges, gt)
            mask = owner >= 0
            fallbacks += fallback
            gt_count += len(gt)
            if len(gt):
                counts = torch.bincount(owner[mask], minlength=len(gt))
                per_gt.extend(counts.cpu().tolist())
            if mask.any():
                classification[batch, mask, target['labels'][owner[mask]]] = 1
                positive_boxes.append(boxes[batch, mask])
                target_boxes.append(gt[owner[mask]])
        positives = sum(len(x) for x in positive_boxes)
        normalizer = max(positives, 1)
        losses = {'loss_dense_cls': sigmoid_focal_loss(logits, classification, alpha=.25, gamma=2,
                                                      reduction='sum') / normalizer}
        if positives:
            prediction, truth = torch.cat(positive_boxes), torch.cat(target_boxes)
            losses['loss_dense_bbox'] = 5 * F.l1_loss(prediction, truth, reduction='sum') / normalizer
            # Elementwise aligned GIoU avoids quadratic positive-positive matrices.
            a, b = box_cxcywh_to_xyxy(prediction), box_cxcywh_to_xyxy(truth)
            intersection = (torch.minimum(a[:, 2:], b[:, 2:]) -
                            torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(1)
            area_a, area_b = (a[:, 2:] - a[:, :2]).prod(1), (b[:, 2:] - b[:, :2]).prod(1)
            union = (area_a + area_b - intersection).clamp(min=1e-8)
            enclosing = (torch.maximum(a[:, 2:], b[:, 2:]) -
                         torch.minimum(a[:, :2], b[:, :2])).prod(1).clamp(min=1e-8)
            giou = intersection / union - (enclosing - union) / enclosing
            losses['loss_dense_giou'] = 2 * (1 - giou).sum() / normalizer
        else:
            losses['loss_dense_bbox'] = boxes.sum() * 0
            losses['loss_dense_giou'] = boxes.sum() * 0
        self.last_statistics = {'gt_count': gt_count, 'positive_locations': positives,
                                'positives_per_gt': positives / max(gt_count, 1),
                                'zero_positive_gt': sum(x == 0 for x in per_gt),
                                'multi_positive_gt': sum(x > 1 for x in per_gt),
                                'tiny_fallback_gt': fallbacks,
                                'per_gt_counts': per_gt}
        return losses


@register()
class AuxDetectionDFINE(DFINE):
    __inject__ = ['backbone', 'encoder', 'decoder']
    __share__ = ['num_classes']

    def __init__(self, backbone, encoder, decoder, num_classes=12):
        super().__init__(backbone, encoder, decoder)
        self.aux_head = DenseAuxHead(encoder.out_channels[0], num_classes)

    def forward(self, x, targets=None):
        features = self.encoder(self.backbone(x))
        output = self.decoder(features, targets)
        if self.training and targets is not None:
            output['dense_training_losses'] = self.aux_head(features, targets)
        return output


@register()
class AuxDetectionCriterion(DFINECriterion):
    __inject__ = ['matcher']
    __share__ = ['num_classes']

    def __init__(self, matcher, weight_dict, losses, alpha=.2, gamma=2., num_classes=80,
                 reg_max=32, boxes_weight_format=None, share_matched_indices=False, dense_weight=.25):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes, reg_max,
                         boxes_weight_format, share_matched_indices)
        self.dense_weight = dense_weight

    def forward(self, outputs, targets, **kwargs):
        dense_losses = outputs.get('dense_training_losses', {})
        outputs = {k: v for k, v in outputs.items() if k != 'dense_training_losses'}
        result = super().forward(outputs, targets, **kwargs)
        result.update({key: value * self.dense_weight for key, value in dense_losses.items()})
        return result
