"""Read-only COCO matching diagnostic; thresholds describe errors, never tune outputs."""
import contextlib
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser()
parser.add_argument('--predictions-root', type=Path, required=True)
parser.add_argument('--annotations', type=Path, required=True)
args = parser.parse_args()
ROOT = args.predictions_root.resolve()
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(str(args.annotations))
report = {'scope': 'Complete supplied heldout fold; original-area COCO small; unclipped native top100. Standard one-to-one category matching and area ignores. Score thresholds and FP geometry tags are descriptive only; no outputs are changed.', 'annotations_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(), 'arms': {}}
def overlap(box, target):
    x, y, w, h = box
    a, b, c, d = target
    inter = max(0, min(x+w, a+c)-max(x, a))*max(0, min(y+h, b+d)-max(y, b))
    return inter / max(w*h+c*d-inter, 1e-12)

def false_positive_tag(det, image_id, category_id, iou_threshold):
    targets = [a for a in gt.imgToAnns[image_id] if not a.get('iscrowd', 0)]
    same = max([overlap(det['bbox'], a['bbox']) for a in targets if a['category_id'] == category_id], default=0)
    other = max([overlap(det['bbox'], a['bbox']) for a in targets if a['category_id'] != category_id], default=0)
    if same >= iou_threshold:
        return 'overlap_with_already_matched_same_class'
    if same >= .5:
        return 'same_class_iou50_below_eval_threshold'
    if other >= .5:
        return 'other_class_iou50'
    if max(same, other) < .1:
        return 'no_gt_iou10'
    return 'ambiguous_overlap'

reference_ids = None
for arm in ('real', 'sham'):
    pred = json.loads((ROOT / arm / 'raw_predictions.json').read_text())
    ids = set(map(int, pred))
    assert ids == set(gt.imgs), 'Incomplete predictions do not support complete-fold diagnosis'
    assert reference_ids is None or ids == reference_ids
    reference_ids = ids
    rows = []
    for iid, p in pred.items():
        for b, s, c in zip(p['boxes'], p['scores'], p['labels']):
            rows.append(dict(image_id=int(iid), category_id=int(c), bbox=[b[0], b[1], b[2]-b[0], b[3]-b[1]], score=s))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(rows)
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(map(int, pred))
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    small_idx = ev.params.areaRngLbl.index('small')
    small_area = ev.params.areaRng[small_idx]
    arm_result = {'map': float(ev.stats[0]*100), 'small_ap': float(ev.stats[3]*100), 'prediction_sha256': hashlib.sha256((ROOT / arm / 'raw_predictions.json').read_bytes()).hexdigest(), 'images': len(ids), 'categories': {}}
    for k, cat in enumerate(ev.params.catIds):
        precision = ev.eval['precision'][:, :, k, small_idx, -1]
        valid = precision[precision >= 0]
        entries = [e for e in ev.evalImgs if e and e['category_id'] == cat and e['aRng'] == small_area]
        item = {'gt_count': sum(int((~np.asarray(e['gtIgnore'], dtype=bool)).sum()) for e in entries), 'small_ap': float(valid.mean()*100) if valid.size else None, 'iou': {}}
        for target in (.5, .75, .9):
            t = int(np.argmin(np.abs(ev.params.iouThrs-target)))
            thresholds = {}
            for threshold in (0, .25, .5):
                tp = fp = 0
                tags = {}
                for e in entries:
                    score = np.asarray(e['dtScores'])
                    matched = np.asarray(e['dtMatches'])[t] > 0
                    keep = (~np.asarray(e['dtIgnore'], dtype=bool)[t]) & (score >= threshold)
                    tp += int((keep & matched).sum())
                    fp += int((keep & ~matched).sum())
                    for index in np.flatnonzero(keep & ~matched):
                        tag = false_positive_tag(dt.anns[e['dtIds'][int(index)]], e['image_id'], cat, target)
                        tags[tag] = tags.get(tag, 0)+1
                assert sum(tags.values()) == fp
                thresholds[str(threshold)] = {'tp': tp, 'fp': fp, 'miss': item['gt_count']-tp, 'fp_geometry_tags': tags}
            item['iou'][str(target)] = thresholds
        arm_result['categories'][gt.cats[cat]['name']] = item
    report['arms'][arm] = arm_result
report['deltas'] = {}
for name, r in report['arms']['real']['categories'].items():
    s = report['arms']['sham']['categories'][name]
    report['deltas'][name] = {'gt_count': r['gt_count'], 'small_ap_delta': r['small_ap']-s['small_ap'] if r['small_ap'] is not None else None, 'iou75_score025': {key: r['iou']['0.75']['0.25'][key]-s['iou']['0.75']['0.25'][key] for key in ('tp', 'fp', 'miss')}}
(ROOT / 'small_matching_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'metrics': {a: {k: report['arms'][a][k] for k in ('map', 'small_ap')} for a in ('real', 'sham')}, 'deltas': report['deltas']}, indent=2))
