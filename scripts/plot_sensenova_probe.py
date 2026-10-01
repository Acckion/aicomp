"""Update local pilot progress every 60s, comparing only fully paired images."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/sensenova_probe'
MONITOR = ROOT / 'monitoring/sensenova_probe'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def generate():
    MONITOR.mkdir(parents=True, exist_ok=True)
    controller, worker, download, summary = [read(OUT/name) for name in
        ['controller_status.json', 'status.json', 'download_status.json', 'summary.json']]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    total = max(download.get('total_chunks', 1), 1)
    done = download.get('chunks', 0)
    ax.barh(['Verified-range download'], [done/total*100], color='#386c9a')
    ax.set_xlim(0, 100); ax.set_xlabel('% of checkpoint ranges downloaded')
    ax.set_title('Official model download; SHA256 checked before inference')
    ax = axes[0, 1]; ax.axis('off')
    lines = [f"Controller: {controller.get('stage', 'preparing')}",
             f"Worker: {worker.get('stage', 'not started')}",
             f"Requests: {worker.get('completed_requests', 0)} / {worker.get('total_requests', 120)}",
             f"Image / variant: {worker.get('image_id', '-')} / {worker.get('variant', '-')}",
             f"Fully paired images: {summary.get('paired_images', 0)} / 24",
             f"GPU7 physical cards: {controller.get('gpus', [0, 1, 2, 3])}; 8.5 GiB PyTorch cap each",
             'Persistent storage: GPU6 /home2/fbohan/AIC_storage',
             'No phase2 inference or fitting; zero-shot validation pilot']
    error = worker.get('error') or controller.get('error')
    if error:
        lines += ['ERROR: '+error[:170]]
    if summary.get('results'):
        failures = {name: data.get('response_format_errors', 0)
                    for name, data in summary['results'].items() if name != 'dfine_reference'}
        lines += ['Format failures: '+str(failures)]
    ax.text(0, 1, '\n\n'.join(lines), va='top', fontsize=9)
    labels = list(summary.get('results', {}))
    for ax, key, title in [(axes[1, 0], 'map50_95', 'mAP@50-95: uncalibrated token likelihood vs native D-FINE scores'),
                           (axes[1, 1], 'f1', 'One-to-one F1 @ IoU .75: geometry diagnostic')]:
        if labels:
            values = []
            for label in labels:
                result = summary['results'][label]
                if key == 'f1':
                    values.append(result['one_to_one_geometry']['0.75']['f1']*100)
                else:
                    scores = result.get('token_likelihood_proxy', result.get('native_scores', {}))
                    values.append(scores.get(key, 0))
            ax.bar(range(len(labels)), values, color=['#3b729b', '#b28335', '#6c9955', '#898989', '#986888', '#4b5055'])
            ax.set_xticks(range(len(labels)), labels, rotation=22, ha='right', fontsize=8)
        else:
            ax.text(.5, .5, 'Waiting for first complete paired image\nNo performance conclusion yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10); ax.set_ylabel('0-100'); ax.set_ylim(0, 100)
    figure.suptitle('SenseNova-Vision RGB / IR pilot; same fully paired subset', fontsize=15)
    figure.text(.01, .01, datetime.now().astimezone().isoformat(timespec='seconds')+
                ' | Refresh: 60s | Small changing cohort; not full-val or phase2 score', fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, .94))
    temp = MONITOR / 'overview.tmp.png'; figure.savefig(temp, dpi=120)
    temp.replace(MONITOR/'overview.png'); plt.close(figure)
    return controller.get('stage') in ['complete', 'failed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        terminal = generate()
        if terminal or not args.watch:
            break
        time.sleep(60)
