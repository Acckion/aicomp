"""Compare completed mechanism controls; never automatically promote to full data."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/mechanism_followup'
NAMES = ['bn_adapt800', 'bn_frozen800', 'relative_box800', 'mal800', 'dense_o2o800']
ENV = {**os.environ, 'OMP_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '7',
       'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def status(stage, **details):
    write_json(OUT / 'status.json', {'stage': stage, 'time': time.time(), **details})


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(name):
    path = ROOT / 'runs' / name / 'metrics.jsonl'
    rows = [json.loads(s) for s in path.read_bytes().splitlines(keepends=True)
            if s.endswith(b'\n')] if path.exists() else []
    complete = (ROOT / 'runs' / name / 'COMPLETE').exists()
    if complete and len(rows) != 12:
        raise RuntimeError(f'{name}: COMPLETE without 12 metrics rows')
    result = {'completed_epochs': len(rows), 'complete': complete,
              'initial_validation': read_json(ROOT / 'runs' / name / 'initial_metrics.json')}
    if not complete:
        result['failure'] = read_json(ROOT / 'experiments/mechanism_trials' / f'{name}_status.json')
        return result
    result['median_last5'] = {
        'map': statistics.median(r['validation']['coco_eval_bbox'][0] * 100 for r in rows[-5:]),
        'ap75': statistics.median(r['validation']['coco_eval_bbox'][2] * 100 for r in rows[-5:]),
        'small_ap': statistics.median(r['validation']['coco_eval_bbox'][3] * 100 for r in rows[-5:]),
        'ap90': statistics.median(r['validation']['ap_by_iou']['0.90'] * 100 for r in rows[-5:]),
    }
    result['per_class_median_last5'] = {
        c: statistics.median(r['validation']['per_class_ap'][c] * 100 for r in rows[-5:])
        for c in rows[0]['validation']['per_class_ap']
    }
    result['best_recorded_training_map_epoch'] = max(rows, key=lambda r: r['validation']['coco_eval_bbox'][0])['epoch']
    return result


def auxiliary_cohorts(folder):
    from faster_coco_eval import COCO
    from evaluate_variants import evaluate, select
    data = read_json(ROOT / 'data/annotations/val400.json')
    by_image = {i['id']: [] for i in data['images']}
    for annotation in data['annotations']:
        by_image[annotation['image_id']].append(annotation)
    short_heavy = []
    for image in data['images']:
        boxes = by_image[image['id']]
        short = sum(min(a['bbox'][2] * 800 / image['width'],
                        a['bbox'][3] * 800 / image['height']) < 16 for a in boxes)
        if len(boxes) >= 3 and short / len(boxes) >= .5:
            short_heavy.append(image['id'])
    similarity = read_json(ROOT / 'experiments/mechanism_audit/split_similarity.json')
    near = {p['val_image_id'] for p in similarity.get('pairs', [])
            if p['phash_hamming63'] <= 4 and p['rgb_rmse32'] <= .05}
    cohorts = {'short_side_under16_heavy': short_heavy}
    if similarity:
        cohorts['excluding_near_frame_candidates'] = [i for i in by_image if i not in near]
    raw = read_json(folder / 'raw_predictions.json')
    gt = COCO(str(ROOT / 'data/annotations/val400.json'))
    output = {}
    with (folder / 'auxiliary_cohort_evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
        for label, ids in cohorts.items():
            if ids:
                output[label] = {'images': len(ids), 'gt_boxes': sum(len(by_image[i]) for i in ids),
                                 **evaluate(gt, {str(i): select(raw[str(i)]) for i in ids})}
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        terminal = [(ROOT / 'runs' / n / 'COMPLETE').exists() or
                    read_json(ROOT / 'experiments/mechanism_trials' / f'{n}_status.json').get('stage') == 'failed'
                    for n in NAMES]
        if all(terminal):
            break
        status('waiting_for_training', terminal=dict(zip(NAMES, terminal)))
        time.sleep(30)
    report = {'runs': {n: summarize(n) for n in NAMES}, 'deltas': {},
              'note': 'Same train1600/val400; last-five operational comparison is not statistical significance or phase2 performance. No full-data promotion.'}
    for name in NAMES:
        control = 'bn_adapt800' if name == 'bn_frozen800' else 'bn_frozen800'
        if name == 'bn_adapt800' or not all(report['runs'][n]['complete'] for n in (name, control)):
            continue
        current, reference = (report['runs'][n] for n in (name, control))
        report['deltas'][name] = {'control': control,
            'median_last5_delta': {k: v - reference['median_last5'][k] for k, v in current['median_last5'].items()},
            'per_class_median_last5_delta': {k: v - reference['per_class_median_last5'][k]
                                           for k, v in current['per_class_median_last5'].items()}}
    write_json(OUT / 'comparison.json', report)
    for name in NAMES:
        if not report['runs'][name]['complete']:
            continue
        while int(subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'], text=True, timeout=10).strip()) < 9900:
            status('waiting_for_evaluation_memory', run=name, gpu=7)
            time.sleep(30)
        folder = OUT / name
        status('evaluating_best', run=name, gpu=7)
        command = [sys.executable, str(ROOT / 'scripts/evaluate_variants.py'),
                   '--checkpoint', str(ROOT / 'runs' / name / 'best.pth'),
                   '--config', str(ROOT / 'configs/ft_aug800_shared3.yml'), '--size', '800',
                   '--output', str(folder), '--single-method', '--method', 'none',
                   '--gpu-memory-limit-gib', '8.5']
        with (OUT / 'evaluation.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/diagnose_validation.py'),
                            '--predictions', str(folder / 'raw_predictions.json'),
                            '--output', str(folder / 'diagnostics'), '--source', f'{name} saved best EMA, 800 plain'],
                           cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
        report['runs'][name]['best_plain800_cohorts'] = read_json(folder / 'diagnostics/report.json')['cohorts']
        report['runs'][name]['best_auxiliary_cohorts'] = auxiliary_cohorts(folder)
        write_json(OUT / 'comparison.json', report)
    status('comparison_ready', report=str(OUT / 'comparison.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        status('failed', traceback=traceback.format_exc())
        raise
