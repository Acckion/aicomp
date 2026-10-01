"""Refresh four new jobs every 60 seconds; full-data jobs have no validation AP."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
from plot_training import style
from run_next_mechanisms import JOBS, ROOT, OUT as STATE, write_json

OUT = ROOT / 'monitoring/next_mechanisms'
LABELS = {'bn_frozen800': '1600 冻结BN对照', 'highorder800': '1600 高质量匹配',
          'queryrank800': '1600 匹配＋候选排序', 'full2000_bn800': '2000 冻结BN',
          'full2000_dense800': '2000 冻结BN＋密集增强'}
COLORS = ['#73814d', '#2766a2', '#a3487e', '#c07832', '#6d5690']


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def rows(name):
    path = ROOT / 'runs' / name / 'metrics.jsonl'
    if not path.exists():
        return []
    result = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
              if line.endswith(b'\n')]
    if [r['epoch'] for r in result] != list(range(1, len(result) + 1)):
        raise RuntimeError(f'Duplicate/missing epochs: {name}')
    return result


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: rows(name) for name in LABELS}
    status = {'updated_at': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
              'free_disk_gib': shutil.disk_usage(ROOT).free / 1024**3, 'runs': {}}
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panels = [(0, 'mAP@50–95 · 独立val400'), (2, 'AP75 · 独立val400'),
              ('ap90', 'AP90 · 独立val400'), (3, '小目标AP · 独立val400')]
    for ax, (index, title) in zip(axes[0], panels):
        for (name, label), color in zip(LABELS.items(), COLORS):
            if name.startswith('full2000'):
                continue
            initial = read_json(ROOT / 'runs' / name / 'initial_metrics.json')
            points = []
            if initial:
                value = initial.get('ap_by_iou', {}).get('0.90') if index == 'ap90' else initial['coco_eval_bbox'][index]
                if value is not None:
                    points.append((0, value * 100))
            for r in data[name]:
                v = r['validation']
                value = v.get('ap_by_iou', {}).get('0.90') if index == 'ap90' else v['coco_eval_bbox'][index]
                if value is not None:
                    points.append((r['epoch'], value * 100))
            if points:
                ax.plot(*zip(*points), label=label, color=color, marker='.', lw=1.6)
        style(ax, title, 'AP（0–100）')
        ax.set_xlim(0, 12)
        if ax.lines:
            ax.legend(fontsize=8)
    for (name, label), color in zip(LABELS.items(), COLORS):
        x = [r['epoch'] for r in data[name]]
        if name.startswith('full2000'):
            axes[1, 0].plot(x, [r['train']['loss'] for r in data[name]], label=label, color=color, marker='.')
        elif name != 'bn_frozen800':
            axes[1, 1].plot(x, [r['train'].get('loss_query_rank', 0) for r in data[name]],
                            label=label, color=color, marker='.')
        axes[1, 2].plot(x, [r['lr'][-1] for r in data[name]], label=label, color=color)
    style(axes[1, 0], '2000全量训练损失（不等于验证效果）', '加权损失')
    style(axes[1, 1], '额外候选排序损失（匹配组无此损失）', '加权损失')
    style(axes[1, 2], '动态学习率 · 检测头', '学习率')
    for ax in axes[1, :3]:
        if ax.lines:
            ax.legend(fontsize=8)
    lines = []
    for name, job in JOBS.items():
        controller = read_json(STATE / f'{name}_status.json')
        progress = read_json(ROOT / 'runs' / name / 'optimizer_progress.json')
        complete = (ROOT / 'runs' / name / 'COMPLETE').exists() and len(data[name]) == 12
        status['runs'][name] = {'completed_epochs': len(data[name]), 'planned_epochs': 12,
                               'complete': complete, 'controller': controller, 'progress': progress,
                               'validation': job['validation']}
        detail = f"epoch {progress.get('epoch')} / batch {progress.get('next_batch')}" if progress else controller.get('stage', '准备中')
        lines.append(f"{LABELS[name]} · GPU{job['gpu']}\n{len(data[name])}/12轮 · {detail}")
    axes[1, 3].axis('off')
    axes[1, 3].text(0, 1, '\n\n'.join(lines), va='top', fontsize=10)
    fig.suptitle('D-FINE-X · 全量迁移与高质量候选机制实验', fontsize=18)
    fig.text(.02, .02, f"{status['updated_at']} | 每60秒更新 | 完整epoch才产生AP | 2000版不使用训练集AP挑权重 | 磁盘余量 {status['free_disk_gib']:.2f} GiB", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    temporary = OUT / 'overview.tmp.png'
    fig.savefig(temporary, dpi=130)
    temporary.replace(OUT / 'overview.png')
    plt.close(fig)
    write_json(OUT / 'status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'plot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        report = generate()
        terminal = [r['complete'] or r['controller'].get('stage') == 'failed' for r in report['runs'].values()]
        if not args.watch or all(terminal):
            break
        time.sleep(60)
