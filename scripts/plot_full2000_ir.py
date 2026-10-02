"""Minute-level full-data monitoring; no independent validation AP exists."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from plot_training import style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring/full2000_ir'
RUN = ROOT / 'runs/full2000_ir_content800'
EXP = ROOT / 'experiments/full2000_ir'


def read(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def generate():
    records = []
    try:
        for line in (RUN / 'metrics.jsonl').read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'): continue
            try: records.append(json.loads(line))
            except ValueError: pass
    except OSError: pass
    state = read(EXP / 'status.json')
    current = ''
    try:
        path = EXP / 'full2000_ir_content800.log'
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 24000))
            text = f.read().decode(errors='replace')
        matches = re.findall(r'Epoch: \[(\d+)/8\]\s+\[\s*(\d+)/(\d+)\].*?loss: ([\d.]+) \(([\d.]+)\)', text)
        if matches:
            epoch, batch, total, _, loss = matches[-1]
            current = f'第{int(epoch)+1}/8轮 · {batch}/{total}批次\n本轮累计平均loss：{loss}'
    except OSError: pass
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in [(axes[0, 0], 'loss', '每轮训练loss'),
                              (axes[0, 1], 'loss_bbox', '每轮框回归loss')]:
        pairs = [(r['epoch'], r['train'][metric]) for r in records if metric in r.get('train', {})]
        if pairs: ax.plot(*zip(*pairs), marker='o')
        else: ax.text(.5, .5, '尚无完成轮次', ha='center', transform=ax.transAxes)
        style(ax, title); ax.set_xlabel('Epoch')
    ax = axes[1, 0]
    if records:
        count = max(len(r.get('lr', [])) for r in records)
        for i in range(count):
            pairs = [(r['epoch'], r['lr'][i]) for r in records if len(r.get('lr', [])) > i]
            if pairs: ax.plot(*zip(*pairs), marker='o', label=f'参数组{i+1}')
        ax.legend(); ax.set_yscale('log')
    ax.set_title('动态学习率'); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[1, 1].axis('off')
    updated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    lines = ['2000张官方训练图 · RGB＋IR内容对齐',
             'GPU7服务器物理卡3 · 显存上限8.5GiB',
             f"状态：{state.get('stage', '准备中')}",
             f'已完成：{len(records)}/8轮', current, '',
             '全量训练没有独立验证集，不展示验证mAP。',
             '每轮保留权重；轮次由独立1600实验预先参考。',
             f'每60秒刷新 · {updated}']
    axes[1, 1].text(0, 1, '\n'.join(lines), va='top', fontsize=10)
    fig.suptitle('AICOMP · 全量2000张红外内容对齐', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    tmp = OUT / f'overview.{os.getpid()}.tmp.png'
    fig.savefig(tmp, dpi=120); tmp.replace(OUT / 'overview.png'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--watch', action='store_true'); args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'monitor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        generate()
        if not args.watch: break
        time.sleep(60)
