"""Refresh source-backed mechanism trial curves and progress every 60 seconds."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plot_targetcrop as charts
from plot_training import style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring/mechanism_trials'
JOBS = {
    'bn_adapt800': ('BN适应对照', '#C47B32', '--', 12, 0),
    'bn_frozen800': ('冻结BN对照', '#697349', '--', 12, 1),
    'relative_box800': ('相对定位监督', '#2F5D9B', '-', 12, 5),
    'mal800': ('MAL分类监督', '#A9557D', '-', 12, 6),
    'dense_o2o800': ('局部四图密集增强', '#705B89', '-', 12, 4),
}
charts.OUT = OUT
charts.RUNS = {name: item[:4] for name, item in JOBS.items()}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def generate():
    charts.generate()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [('0.75', 'AP75'), ('0.90', 'AP90'), ('0.95', 'AP95')]
    status = {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'free_disk_gib': shutil.disk_usage(ROOT).free / 1024**3, 'runs': {}}
    for name, (label, color, line, epochs, gpu) in JOBS.items():
        rows, _ = charts.read_rows(ROOT / 'runs' / name / 'metrics.jsonl')
        initial = read_json(ROOT / 'runs' / name / 'initial_metrics.json')
        progress = read_json(ROOT / 'runs' / name / 'optimizer_progress.json')
        state = read_json(ROOT / 'experiments/mechanism_trials' / f'{name}_status.json')
        for axis, (threshold, title) in zip(axes.flat, panels):
            points = [(r['epoch'], r['validation']['ap_by_iou'][threshold] * 100)
                      for r in rows if r['validation'].get('ap_by_iou', {}).get(threshold) is not None]
            if initial.get('ap_by_iou', {}).get(threshold) is not None:
                points.insert(0, (0, initial['ap_by_iou'][threshold] * 100))
            if points:
                axis.plot(*zip(*points), color=color, ls=line, marker='o', ms=3, label=label)
        if rows:
            relative = [sum(v for k, v in r['train'].items() if k.startswith('loss_relative_box')) for r in rows]
            if name == 'relative_box800':
                axes.flat[3].plot([r['epoch'] for r in rows], relative, color=color, marker='o', label=label)
        status['runs'][name] = {
            'gpu': gpu, 'completed_epochs': len(rows), 'planned_epochs': epochs,
            'complete': (ROOT / 'runs' / name / 'COMPLETE').exists(),
            'controller': state, 'training_pid_alive': pid_alive(state.get('pid')),
            'optimizer_progress': progress,
            'latest_ap90': rows[-1]['validation'].get('ap_by_iou', {}).get('0.90') if rows else None,
        }
    for axis, (_, title) in zip(axes.flat, panels):
        style(axis, title + ' · val400', 'AP（0–100）')
    style(axes.flat[3], '新增相对定位损失（各分支加权总和）', '损失')
    for axis in axes.flat[:4]:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(0, max(2, max(r['completed_epochs'] for r in status['runs'].values())))
        if axis.lines:
            axis.legend(fontsize=8)
    for axis in axes.flat[:3]:
        if axis.lines:
            values = [v for line in axis.lines for v in line.get_ydata()]
            low, high = min(values), max(values)
            if high - low < 2:
                center = (low + high) / 2
                axis.set_ylim(max(0, center - 1), min(100, center + 1))
    for axis in axes.flat[4:]:
        axis.axis('off')
    text = []
    for name, info in status['runs'].items():
        p = info['optimizer_progress']
        detail = f"轮内epoch {p.get('epoch', '?')} / batch {p.get('next_batch', '?')}" if p else info['controller'].get('stage', '准备中')
        text.append(f"{JOBS[name][0]} · GPU{info['gpu']} · {info['completed_epochs']}/12轮 · {detail}")
    axes.flat[4].text(0, .95, '\n\n'.join(text), va='top', fontsize=10)
    axes.flat[5].text(0, .95,
        f"北京时间 {status['updated_at'][:19]}\n\n磁盘可用 {status['free_disk_gib']:.2f} GiB\n\n"
        "每60秒刷新；AP仅来自完整验证。\n不同损失的数值不能直接判断模型优劣。\n"
        "原始训练评测与裁剪后提交评测分开报告。\n保存每轮指标、最佳/最新全状态，\n额外推理权重保留第3/6/9/12轮。\n本地指标不代表phase2成绩。", va='top', fontsize=10)
    fig.suptitle('D-FINE-X · 高IoU定位与训练机制对照', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, .94))
    temp = OUT / 'high_iou.tmp.png'
    fig.savefig(temp, dpi=145)
    temp.replace(OUT / 'high_iou.png')
    plt.close(fig)
    temp = OUT / 'progress.tmp.json'
    temp.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    temp.replace(OUT / 'progress.json')
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
