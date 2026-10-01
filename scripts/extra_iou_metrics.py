"""Expose per-IoU AP from the existing evaluator without another inference pass."""
import numpy as np
import train_baseline as baseline


def _valid_mean(values):
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def add_iou_metrics(stats, evaluator):
    coco = evaluator.coco_eval.get('bbox') if evaluator is not None else None
    if coco is None or not coco.eval:
        return stats
    precision = coco.eval['precision']
    # COCO dimensions: IoU, recall, category, area, max detections.
    stats['ap_by_iou'] = {
        f'{threshold:.2f}': _valid_mean(precision[index, :, :, 0, -1])
        for index, threshold in enumerate(coco.params.iouThrs)
    }
    found = np.flatnonzero(np.isclose(coco.params.iouThrs, .90))
    if len(found):
        index = int(found[0])
        stats['per_class_ap90'] = {
            evaluator.coco_gt.cats[category]['name']: _valid_mean(precision[index, :, k, 0, -1])
            for k, category in enumerate(coco.params.catIds)
        }
    return stats


if not getattr(baseline.evaluate, '_extra_iou_metrics', False):
    _original_evaluate = baseline.evaluate

    def evaluate(*args, **kwargs):
        stats, evaluator = _original_evaluate(*args, **kwargs)
        return add_iou_metrics(stats, evaluator), evaluator

    evaluate._extra_iou_metrics = True
    baseline.evaluate = evaluate
