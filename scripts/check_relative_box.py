"""Focused numerical checks for the new localization mechanism, no dataset."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
import torch
from relative_box_criterion import RelativeBoxCriterion, relative_box_loss


class FirstMatcher:
    def __call__(self, outputs, targets):
        return {'indices': [(torch.arange(len(t['boxes'])),
                            torch.arange(len(t['boxes']))) for t in targets]}


def main():
    target = torch.tensor([[.5, .5, .02, .02], [.5, .5, .2, .2]])
    pred = target.clone().requires_grad_()
    assert relative_box_loss(pred, target, 2).item() == 0
    shifted = target.clone()
    shifted[:, 0] += 2 / 800
    small = relative_box_loss(shifted[:1], target[:1], 1)
    large = relative_box_loss(shifted[1:], target[1:], 1)
    assert small > large > 0
    tiny = torch.tensor([[.5, .5, 1e-9, 1e-9]])
    extreme = torch.tensor([[.99, .99, .9, .9]], requires_grad=True)
    extreme_loss = relative_box_loss(extreme, tiny, 1)
    extreme_loss.backward()
    assert torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all()
    assert extreme.grad.abs().max() <= 1 / .02 / 4 + 1e-6
    empty = torch.empty((0, 4), requires_grad=True)
    empty_loss = relative_box_loss(empty, empty.detach(), 1)
    assert empty_loss.item() == 0
    empty_loss.backward()
    criterion = RelativeBoxCriterion(FirstMatcher(), {'loss_bbox': 5, 'loss_giou': 2},
                                     ['boxes'], num_classes=12)
    boxes = shifted.unsqueeze(0).clone().requires_grad_()
    def head():
        return {'pred_boxes': boxes, 'pred_logits': torch.zeros(1, 2, 12)}
    outputs = {**head(), 'aux_outputs': [head()], 'pre_outputs': head(),
               'enc_aux_outputs': [head()], 'enc_meta': {'class_agnostic': False},
               'dn_outputs': [head()], 'dn_pre_outputs': head(),
               'dn_meta': {'dn_positive_idx': [torch.arange(2)], 'dn_num_group': 1},
               'up': torch.tensor(1.), 'reg_scale': torch.tensor(4.)}
    targets = [{'boxes': target, 'labels': torch.tensor([0, 1])}]
    losses = criterion(outputs, targets)
    relative_keys = [k for k in losses if k.startswith('loss_relative_box')]
    assert len(relative_keys) == 6, relative_keys
    expected = relative_box_loss(shifted, target, 2) * .25
    assert all(torch.allclose(losses[k], expected) for k in relative_keys)
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(boxes.grad).all()
    zero_boxes = torch.empty((1, 0, 4), requires_grad=True)
    no_gt = [{'boxes': torch.empty((0, 4)), 'labels': torch.empty(0, dtype=torch.long)}]
    zero_head = {'pred_boxes': zero_boxes, 'pred_logits': torch.empty((1, 0, 12))}
    zero_outputs = {**zero_head, 'aux_outputs': [], 'pre_outputs': dict(zero_head),
                    'enc_aux_outputs': [], 'enc_meta': {'class_agnostic': False},
                    'up': torch.tensor(1.), 'reg_scale': torch.tensor(4.)}
    zero_losses = criterion(zero_outputs, no_gt)
    assert all(torch.isfinite(x) and x.item() == 0 for x in zero_losses.values())
    print(json.dumps({'checks': 'passed', 'small_shift_unweighted': small.item(),
                      'large_shift_unweighted': large.item(),
                      'relative_branch_keys': relative_keys,
                      'extreme_gradient_max': extreme.grad.abs().max().item(),
                      'all_empty_zero': True}, indent=2))


if __name__ == '__main__':
    main()
