"""Finish screening and package full-data candidates without claiming test AP."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

from run_next_mechanisms import JOBS, ROOT, OUT as STATE, ENV, write_json
from plot_next_mechanisms import rows, read_json
from after_mechanism_experiments import auxiliary_cohorts

OUT = ROOT / 'experiments/next_mechanism_followup'


def summarize(name):
    data = rows(name)
    result = {'completed_epochs': len(data), 'complete': (ROOT / 'runs' / name / 'COMPLETE').exists(),
              'validation': JOBS.get(name, {}).get('validation', True)}
    if not result['validation'] or not data:
        return result
    result['median_last5'] = {
        'map': statistics.median(r['validation']['coco_eval_bbox'][0] * 100 for r in data[-5:]),
        'ap75': statistics.median(r['validation']['coco_eval_bbox'][2] * 100 for r in data[-5:]),
        'ap90': statistics.median(r['validation']['ap_by_iou']['0.90'] * 100 for r in data[-5:]),
        'small_ap': statistics.median(r['validation']['coco_eval_bbox'][3] * 100 for r in data[-5:])}
    result['per_class_median_last5'] = {
        c: statistics.median(r['validation']['per_class_ap'][c] * 100 for r in data[-5:])
        for c in data[0]['validation']['per_class_ap']}
    result['saved_best_epoch'] = data[-1]['best_epoch']
    return result


def main(package_saved_candidates=False, gpu=3):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **kwargs):
        write_json(OUT / 'status.json', {'stage': stage, 'time': time.time(), **kwargs})
    while True:
        terminal = {n: (ROOT / 'runs' / n / 'COMPLETE').exists() or
                    read_json(STATE / f'{n}_status.json').get('stage') == 'failed' for n in JOBS}
        if all(terminal.values()):
            break
        status('waiting_for_training', terminal=terminal)
        time.sleep(30)
    report = {'runs': {n: summarize(n) for n in ['bn_frozen800', *JOBS]}, 'deltas': {},
              'packages': {}, 'errors': {},
              'note': '1600/400 screening only. Full-data models have no independent validation. No automatic promotion of new matching/ranking mechanisms.'}
    for name, control in [('highorder800', 'bn_frozen800'), ('queryrank800', 'highorder800')]:
        if all(report['runs'][n]['complete'] for n in (name, control)):
            current, reference = [report['runs'][n]['median_last5'] for n in (name, control)]
            report['deltas'][name] = {'control': control, **{k: current[k] - reference[k] for k in current}}
    write_json(OUT / 'comparison.json', report)

    def run_command(command, stage, name):
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
            if free >= 9900:
                break
            status('waiting_for_memory', run=name, gpu=gpu)
            time.sleep(10)
        status(stage, run=name, gpu=gpu)
        with (OUT / 'evaluation.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    # Full-data epoch choices are fixed by the previous independent experiment:
    # frozen-BN best epoch 2, dense-view best epoch 3. No train-set AP selection.
    for name, epoch in [('full2000_dense800', 3), ('full2000_bn800', 2)]:
        if not report['runs'][name]['complete'] and not package_saved_candidates:
            continue
        try:
            folder = OUT / f'{name}_epoch{epoch}_phase2'
            checkpoint = ROOT / 'runs' / name / f'weights_epoch_{epoch:03d}.pth'
            if len(rows(name)) < epoch or not checkpoint.exists():
                raise RuntimeError('Selected epoch did not complete or its weights are missing')
            command = [sys.executable, str(ROOT / 'scripts/evaluate_variants.py'),
                       '--checkpoint', str(checkpoint), '--config', str(ROOT / 'configs/ft_aug800_shared3.yml'),
                       '--size', '800', '--method', 'none', '--predict-only',
                       '--annotations', str(ROOT / 'data/phase2/images.json'),
                       '--image-root', str(ROOT / 'data/phase2'), '--output', str(folder)]
            run_command(command, 'packaging_phase2', name)
            archive = folder / 'submission.zip'
            expected = {Path(i['file_name']).stem + '.txt' for i in read_json(ROOT / 'data/phase2/images.json')['images']}
            with zipfile.ZipFile(archive) as z:
                assert set(z.namelist()) == expected, 'Missing/unexpected submission filenames'
                assert z.testzip() is None
            report['packages'][name] = {'epoch': epoch, 'zip': str(archive), 'images': len(expected),
                                        'source_training_complete': report['runs'][name]['complete'],
                                        'source_completed_epochs': report['runs'][name]['completed_epochs'],
                                        'selection': 'Fixed epoch from independent train1600/val400 experiment',
                                        'inference': '800 full RGB image, native top100, no flip TTA or Soft-NMS'}
        except Exception:
            report['errors'][name] = traceback.format_exc()
        write_json(OUT / 'comparison.json', report)
    for name in ['highorder800', 'queryrank800']:
        if not report['runs'][name]['complete']:
            continue
        try:
            folder = OUT / name
            command = [sys.executable, str(ROOT / 'scripts/evaluate_variants.py'),
                       '--checkpoint', str(ROOT / 'runs' / name / 'best.pth'),
                       '--config', str(ROOT / 'configs/ft_aug800_shared3.yml'), '--size', '800',
                       '--output', str(folder), '--single-method', '--method', 'none']
            run_command(command, 'evaluating_best', name)
            run_command([sys.executable, str(ROOT / 'scripts/diagnose_validation.py'),
                         '--predictions', str(folder / 'raw_predictions.json'),
                         '--output', str(folder / 'diagnostics'), '--source', f'{name} saved best, 800 plain'],
                        'evaluating_cohorts', name)
            report['runs'][name]['best_plain800_cohorts'] = read_json(folder / 'diagnostics/report.json')['cohorts']
            report['runs'][name]['best_auxiliary_cohorts'] = auxiliary_cohorts(folder)
        except Exception:
            report['errors'][name] = traceback.format_exc()
        write_json(OUT / 'comparison.json', report)
    status('complete_with_errors' if report['errors'] else 'comparison_ready', report=str(OUT / 'comparison.json'))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--package-saved-candidates', action='store_true',
                            help='Recover completed selected epochs even if later training failed')
        parser.add_argument('--gpu', type=int, default=3)
        args = parser.parse_args()
        main(args.package_saved_candidates, args.gpu)
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / 'status.json', {'stage': 'failed', 'traceback': traceback.format_exc()})
        raise
