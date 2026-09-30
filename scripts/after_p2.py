"""Wait for P2 completion and compare plain inference on identical val cohorts."""
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
OUT = ROOT / 'experiments/p2_followup'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '7', 'OMP_NUM_THREADS': '2',
       'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def status(stage, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / 'status.tmp'
    tmp.write_text(json.dumps({'stage': stage, 'time': time.strftime('%F %T'), **details}, indent=2))
    tmp.replace(OUT / 'status.json')


def completed_rows():
    rows = [json.loads(line) for line in (ROOT / 'runs/p2_640/metrics.jsonl').read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
    assert [r['epoch'] for r in rows] == list(range(1, 21)), 'Expected 20 complete epochs'
    return rows


def summarize(rows):
    best = max(rows, key=lambda r: r['validation']['coco_eval_bbox'][0])
    return {'best_epoch': best['epoch'],
            'best_map': best['validation']['coco_eval_bbox'][0] * 100,
            'last5_median_map': statistics.median(r['validation']['coco_eval_bbox'][0] * 100 for r in rows[-5:]),
            'last5_median_small_ap': statistics.median(r['validation']['coco_eval_bbox'][3] * 100 for r in rows[-5:])}


def wait_memory():
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--id=7', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
        if free >= 9900:
            return
        time.sleep(10)


def run(cmd, logfile):
    with logfile.open('a') as log:
        subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('waiting_for_p2_completion')
    while not (ROOT / 'runs/p2_640/COMPLETE').exists():
        state_file = ROOT / 'experiments/p2/status.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get('stage') == 'failed':
            raise RuntimeError('P2 training failed; no automatic restart')
        if state.get('stage') == 'training' and state.get('pid'):
            try:
                os.kill(state['pid'], 0)
            except ProcessLookupError:
                # Allow the training controller to finish its state update.
                time.sleep(5)
                if not (ROOT / 'runs/p2_640/COMPLETE').exists():
                    raise RuntimeError('P2 training exited without COMPLETE')
        time.sleep(5)
    rows = completed_rows()
    training = summarize(rows)
    initial = json.loads((ROOT / 'runs/p2_640/initial_metrics.json').read_text())
    training['initial_map'] = initial['coco_eval_bbox'][0] * 100
    epoch = training['best_epoch']
    candidates = {
        'baseline': (ROOT / 'runs/ft_aug800/weights_epoch_020.pth', None),
        'p2': (ROOT / f'runs/p2_640/weights_epoch_{epoch:03d}.pth', ROOT / 'configs/p2_640.yml'),
    }
    reports = {}
    for size in (640, 800):
        for name, (weights, config) in candidates.items():
            folder = OUT / f'{name}_{size}'
            wait_memory()
            status('evaluating', model=name, size=size, training=training)
            command = [ROOT / 'scripts/evaluate_variants.py', '--checkpoint', weights,
                       '--size', size, '--output', folder, '--single-method', '--method', 'none']
            if config:
                command += ['--config', config]
            run(command, OUT / 'evaluation.log')
            run([ROOT / 'scripts/diagnose_validation.py', '--predictions', folder / 'raw_predictions.json',
                 '--output', folder / 'diagnostics', '--source', f'{name} same-val400 plain inference {size}'], OUT / 'diagnosis.log')
            reports[f'{name}_{size}'] = json.loads((folder / 'diagnostics/report.json').read_text())
    comparisons = {}
    ann = json.loads((ROOT / 'data/annotations/val400.json').read_text())
    counts = {c['name']: sum(a['category_id'] == c['id'] for a in ann['annotations']) for c in ann['categories']}
    for size in (640, 800):
        base, p2 = reports[f'baseline_{size}'], reports[f'p2_{size}']
        deltas = {name: p2['cohorts']['all']['per_class'][name] - base['cohorts']['all']['per_class'][name] for name in counts}
        comparisons[str(size)] = {
            'cohort_map_delta': {c: p2['cohorts'][c]['map'] - base['cohorts'][c]['map'] for c in base['cohorts']},
            'small_ap_delta': p2['cohorts']['all']['stats'][3] - base['cohorts']['all']['stats'][3],
            'per_class_delta': deltas,
            'common_class_drops_over2': [name for name in counts if counts[name] >= 100 and deltas[name] < -2],
        }
    report = {'training': training, 'comparisons_at_matching_resolution': comparisons,
              'notes': ['Single-model plain inference; no phase2 labels, pseudo-labels or training.',
                        'Best epoch uses val400: selection bias remains. Last5 medians are stability evidence, not significance.',
                        'P2 also changes BN handling and training resolution; this is a candidate comparison, not pure architectural attribution.',
                        'Overlapping cohorts differ in category composition; report sample sizes in each diagnosis.',
                        'No automatic full-data transfer or claim of reaching phase2 57. Formal submission opportunities currently unavailable.']}
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2))
    status('comparison_ready', report=str(OUT / 'comparison.json'), training=training)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise
