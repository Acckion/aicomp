"""Minute-updated fresh pretraining comparison; complete epochs only."""
import fcntl
import json
import os
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring/scene_rgb'
OUT.mkdir(parents=True, exist_ok=True)

def draw():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for name in ['scene_rgb800', 'scene_rgb_fastcontrol800', 'scene_semantic_init800', 'scene_obj365_reset800']:
        path = ROOT / 'runs' / name / 'metrics.jsonl'
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for ax, index, title in zip(list(axes.flat)[:4], [0,1,2,3], ['mAP@50-95','AP50','AP75','Small AP']):
            pairs = [(r['epoch'],100*r['validation']['coco_eval_bbox'][index]) for r in rows if r.get('validation',{}).get('coco_eval_bbox')]
            if pairs:
                ax.plot(*zip(*pairs), marker='o', label=name)
            ax.set_title(title)
        pairs = [(r['epoch'],100*r['validation']['ap_by_iou']['0.90']) for r in rows if '0.90' in r.get('validation',{}).get('ap_by_iou',{})]
        if pairs:
            axes.flat[4].plot(*zip(*pairs), marker='o', label=name)
        pairs = [(r['epoch'],r['train']['loss']) for r in rows if 'loss' in r.get('train',{})]
        if pairs:
            axes.flat[5].plot(*zip(*pairs), marker='o', label=name)
    axes.flat[4].set_title('AP90')
    axes.flat[5].set_title('Loss')
    for ax in axes.flat:
        ax.set_xlabel('epoch')
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle('Fresh RGB: mapped class initialization vs reset | grouped holdout390 | not phase2 | '+time.strftime('%F %T'))
    fig.tight_layout()
    tmp = OUT / f'overview.{os.getpid()}.png'
    fig.savefig(tmp, dpi=130)
    plt.close(fig)
    tmp.replace(OUT / 'overview.png')

if __name__ == '__main__':
    lock = (OUT / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            draw()
        except Exception as error:
            print(repr(error), flush=True)
        time.sleep(60)
