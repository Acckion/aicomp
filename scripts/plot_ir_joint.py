"""Observe the three October 2 experiments using actual persisted metrics."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_training import style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring/ir_joint'
JOBS = {'ir_joint800': ('红外neck联合适配', 'ir_joint', 6, 'ir_joint_control800')}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def rows(run):
    path = ROOT / 'runs' / run / 'metrics.jsonl'
    if run == 'grounding1600' and not path.exists():
        path = ROOT / 'experiments/grounding/metrics.jsonl'
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError:
        return []
    result = []
    for line in lines:
        if not line.endswith(b'\n'):
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def values(run, metric):
    result = []
    initial = read(ROOT / 'runs' / run / 'initial_metrics.json')
    records = ([(0, initial)] if initial else [])
    records += [(r.get('epoch'), r.get('validation', r.get('val', {}))) for r in rows(run)]
    for epoch, data in records:
        if epoch is None or not isinstance(data, dict):
            continue
        stats = data.get('coco_eval_bbox', [])
        if metric == 'ap90':
            fraction = data.get('ap_by_iou', {}).get('0.90')
            if fraction is None:
                fraction = data.get('ap90')
                # Co-DINO explicitly persists AP90 in points.
                if fraction is not None and run == 'codino1600':
                    fraction /= 100
        elif len(stats) > metric:
            fraction = stats[metric]
        else:
            fraction = data.get({0: 'map', 2: 'ap75', 3: 'aps'}[metric])
            if fraction is not None and run == 'codino1600':
                fraction /= 100
        if fraction is not None:
            result.append((epoch, 100 * float(fraction)))
    return result


def generate():
    states = {run: read(ROOT / 'experiments' / folder / 'status.json')
              for run, (_, folder, _, _) in JOBS.items()}
    shared = []
    try:
        query = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        for row in query.splitlines():
            index, used, free, util = map(int, row.split(','))
            shared.append({'gpu': index, 'used_mib': used, 'free_mib': free, 'utilization': util})
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric, title in zip(axes.flat, [0, 2, 'ap90', 3],
                               ['mAP@50–95', 'AP75', 'AP90', '小目标 AP']):
        for i, (run, (label, _, _, control)) in enumerate(JOBS.items()):
            color = plt.get_cmap('tab10')(i)
            data = values(run, metric)
            if data:
                ax.plot(*zip(*data), marker='.', label=label, color=color)
            data = values(control, metric) if control else []
            if data:
                ax.plot(*zip(*data), linestyle='--', label=label + '对照', color=color, alpha=.65)
        reference = values('codino1600', metric)
        if reference:
            ax.plot(*zip(*reference), color='#888888', linestyle=':', label='既有Co-DINO')
        style(ax, title, 'AP（0–100）')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ax.lines:
            ax.legend(fontsize=8)
    for i, (run, (label, _, _, _)) in enumerate(JOBS.items()):
        records = rows(run)
        loss = []
        for r in records:
            value = r.get('train', {}).get('loss', r.get('train_loss'))
            if value is not None:
                loss.append((r['epoch'], value))
        if loss:
            axes[1, 1].plot(*zip(*loss), label=label, color=plt.get_cmap('tab10')(i))
    style(axes[1, 1], '训练损失 · 各实验定义不同', 'Loss')
    if axes[1, 1].lines:
        axes[1, 1].legend(fontsize=8)
    axes[1, 2].axis('off')
    lines = []
    for run, (label, _, gpu, _) in JOBS.items():
        stage = states[run].get('stage', states[run].get('state', '准备中'))
        active_gpu = states[run].get('gpu', gpu)
        lines.append(f'{label}：{stage} · 完成{len(rows(run))}轮 · GPU{active_gpu}')
    lines += ['', '共享显存总占用（包含其他任务）']
    lines += [f"GPU{s['gpu']}：{s['used_mib']/1024:.1f}GiB / 利用率{s['utilization']}%" for s in shared]
    axes[1, 2].text(0, 1, '\n'.join(lines), va='top', fontsize=9)
    updated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    fig.suptitle('AICOMP · 红外neck联合适配与冻结续训对照', fontsize=17)
    fig.text(.02, .012, f'每60秒刷新｜独立val400｜空曲线表示尚无完成评测｜本地分数不等于phase2｜{updated}', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    tmp = OUT / f'overview.{os.getpid()}.tmp.png'
    fig.savefig(tmp, dpi=120)
    tmp.replace(OUT / 'overview.png')
    plt.close(fig)
    temp = OUT / f'status.{os.getpid()}.tmp.json'
    temp.write_text(json.dumps({'updated_at': updated, 'experiments': states,
                                'gpu_shared_totals': shared}, ensure_ascii=False, indent=2))
    temp.replace(OUT / 'status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        generate()
        if not args.watch:
            break
        time.sleep(60)
