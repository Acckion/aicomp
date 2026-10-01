"""Val-only ranking diagnostic: boxes/classes fixed; never export oracle predictions."""
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'D-FINE')]
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import box_iou
from faster_coco_eval import COCO
from evaluate_variants import evaluate, select


def oracle(prediction, annotations, assignment):
    boxes = torch.tensor(prediction['boxes'], dtype=torch.float32).reshape(-1, 4)
    truth = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                           a['bbox'][1] + a['bbox'][3]] for a in annotations], dtype=torch.float32).reshape(-1, 4)
    if not len(boxes) or not len(truth):
        scores = np.zeros(len(boxes))
    else:
        quality = box_iou(boxes, truth).numpy()
        same = np.array(prediction['labels'])[:, None] == np.array([a['category_id'] for a in annotations])[None, :]
        quality = quality * same
        if assignment:
            # Prefer the maximum number of IoU>=.50 matches, then their
            # average success across the official ten thresholds, then IoU.
            success = (quality[:, :, None] >= np.arange(.50, .951, .05)).mean(-1)
            utility = (quality >= .5) * 2 + success + quality * .001
            pi, gi = linear_sum_assignment(utility, maximize=True)
            scores = np.zeros(len(boxes))
            for p, g in zip(pi, gi):
                if quality[p, g] >= .5:
                    scores[p] = quality[p, g]
        else:
            scores = quality.max(1)
    # Exact same coordinates and class IDs; only confidence changes in memory.
    return {'boxes': prediction['boxes'], 'labels': prediction['labels'], 'scores': scores.tolist()}


def main():
    torch.set_num_threads(2)
    out = ROOT / 'experiments/query_pipeline_audit'
    out.mkdir(parents=True, exist_ok=True)
    annotation_path = ROOT / 'data/annotations/val400.json'
    annotations = json.loads(annotation_path.read_text())
    by_image = {str(i['id']): [] for i in annotations['images']}
    for a in annotations['annotations']:
        by_image[str(a['image_id'])].append(a)
    raw = json.loads((ROOT / 'experiments/next_stage/infer800_tile0/raw_predictions.json').read_text())
    assert set(raw) == set(by_image)
    gt = COCO(str(annotation_path))
    report = {'source': 'ft_aug800 epoch20, cached 800 plain, val400', 'results': {},
              'notes': ['Validation oracle only: ground truth used to set scores and optionally disambiguate duplicates.',
                        'Coordinates and class IDs are unchanged. One-to-one assignment is idealized, not an attainable training result.',
                        'These oracles are illustrative, not guaranteed optimal mAP bounds and not additive with TIDE errors.',
                        'No corrected prediction files or submission packages are written. No test data or model gradients.']}
    with (out / 'score_oracle_evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
        original = {i: select(p) for i, p in raw.items()}
        report['results']['native_top100'] = evaluate(gt, original)
        for pool, predictions in [('original_top100', original), ('cached_top300_class_pairs', raw)]:
            for assignment in [False, True]:
                scored = {i: select(oracle(p, by_image[i], assignment)) for i, p in predictions.items()}
                key = f'{pool}_' + ('unique_quality_oracle' if assignment else 'class_iou_score_oracle')
                report['results'][key] = evaluate(gt, scored)
    assert abs(report['results']['native_top100']['map'] - 54.1654768787939) < .01
    (out / 'score_oracle_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {'map': v['map'], 'ap75': v['stats'][2]} for k, v in report['results'].items()}))


if __name__ == '__main__':
    main()
