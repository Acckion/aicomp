"""Numerical checks of assignment quality and overlap-safe ranking gradients."""
import json
import torch
import train_baseline  # register upstream components
from rank_matching import HighOrderMatcher, unique_query_rank_loss


def main():
    matcher = HighOrderMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2})
    target = {'boxes': torch.tensor([[.5, .5, .4, .4]]), 'labels': torch.tensor([0])}
    outputs = {'pred_boxes': torch.tensor([[[.5, .5, .4, .2], [.5, .5, .4, .36]]]),
               'pred_logits': torch.logit(torch.tensor([[[.9], [.6]]]))}
    indices = matcher(outputs, [target])['indices']
    assert indices[0][0].tolist() == [1], 'Prefer high IoU over higher confidence at low IoU'
    topk = matcher(outputs, [target], return_topk=2)['indices_o2m'][0][0]
    assert len(topk.unique()) == 2
    logits = torch.tensor([[[2.0], [0.0]]], requires_grad=True)
    boxes = torch.tensor([[[.5, .5, .4, .4], [.5, .5, .4, .24]]], requires_grad=True)
    out = {'pred_logits': logits, 'pred_boxes': boxes}
    loss, pairs = unique_query_rank_loss(out, [target], [(torch.tensor([0]), torch.tensor([0]))])
    loss.backward()
    assert pairs == 1 and logits.grad[0, 0, 0] < 0 and logits.grad[0, 1, 0] > 0
    assert boxes.grad is None, 'Pair eligibility must not backpropagate geometry'
    protected = {'boxes': torch.tensor([[.5, .5, .4, .4], [.5, .5, .4, .24]]),
                 'labels': torch.tensor([0, 0])}
    loss, pairs = unique_query_rank_loss(out, [protected],
                        [(torch.tensor([0, 1]), torch.tensor([0, 1]))])
    assert pairs == 0 and float(loss) == 0, 'Never suppress another matched overlapping GT'
    loss, pairs = unique_query_rank_loss(out, [target], [(torch.tensor([1]), torch.tensor([0]))])
    assert pairs == 0, 'Do not promote an inferior winner over a more accurate candidate'
    empty = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}
    assert not matcher(out, [empty])['indices'][0][0].numel()
    loss, pairs = unique_query_rank_loss(out, [empty], matcher(out, [empty])['indices'])
    assert pairs == 0 and torch.isfinite(loss) and loss.requires_grad
    print(json.dumps({'assignment_quality': 'pass', 'topk_unique': 'pass',
                      'rank_gradient': 'pass', 'overlap_protection': 'pass',
                      'inferior_winner': 'pass', 'empty_targets': 'pass'}))


if __name__ == '__main__':
    main()
