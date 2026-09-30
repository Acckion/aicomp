"""Snapshot target-view experiment, paired control, and full-data transfer every 20 min."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse, csv, fcntl, hashlib, json, math, time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from plot_training import style, METRICS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring/targetcrop'
RUNS = {
    'targetcrop800': ('小目标裁剪 · 1600', '#2F5D9B', '-', 15),
    'continue800_control': ('普通续训对照 · 1600', '#C47B32', '--', 15),
    'ft2000_aug800': ('全量迁移 · 2000（无独立验证）', '#697349', ':', 20),
}


def read_rows(path):
    raw = path.read_bytes() if path.exists() else b''
    rows = [json.loads(s) for s in raw.splitlines(keepends=True) if s.endswith(b'\n')]
    assert [r['epoch'] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert all(math.isfinite(v) for v in r['train'].values())
        assert all(math.isfinite(v) for v in r['validation'].get('coco_eval_bbox', []))
    return rows, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'epochs': len(rows)}


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    data, sources, initial = {}, {}, {}
    for name in RUNS:
        data[name], sources[name] = read_rows(ROOT / 'runs' / name / 'metrics.jsonl')
        p = ROOT / 'runs' / name / 'initial_metrics.json'
        if p.exists():
            initial[name] = json.loads(p.read_text())['coco_eval_bbox']
    starting_rows, starting_source = read_rows(ROOT / 'runs/ft_aug800/metrics.jsonl')
    reference = next(r for r in starting_rows if r['epoch'] == 20)['validation']
    stamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')

    def lines(ax, getter, validation=False, init_index=None):
        for name, (label, color, ls, planned) in RUNS.items():
            rows = [r for r in data[name] if not validation or r['validation'].get('coco_eval_bbox')]
            x, y = [r['epoch'] for r in rows], [getter(r) for r in rows]
            if init_index is not None and name in initial:
                x, y = [0] + x, [initial[name][init_index] * 100] + y
            if x:
                ax.plot(x, y, color=color, linestyle=ls, marker='o', markersize=3, lw=1.7, label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max(2, max((len(v) for v in data.values()), default=0)))
        if ax.lines:
            ax.legend(fontsize=8, loc='best')

    def save(fig, name, pdf):
        fig.text(.025, .014, f'{stamp} | epoch 0 为开训前实测；对照未启动时不绘制曲线。val400 不代表 phase2。', fontsize=9, color='#555')
        fig.tight_layout(rect=(0, .05, 1, .93))
        temp = OUT / (name + '.tmp.png')
        fig.savefig(temp, dpi=145)
        temp.replace(OUT / (name + '.png'))
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(OUT / 'experiment_metrics.tmp.pdf') as pdf:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        progress = ' / '.join(f'{RUNS[n][0].split(" · ")[0]} {len(data[n])}/{RUNS[n][3]}' for n in RUNS)
        fig.suptitle('D-FINE-X · 本轮实验监控\n' + progress, fontsize=17)
        panels = [(0, 'mAP@50–95'), (1, 'mAP@50'), (3, '小目标 AP'), (4, '中目标 AP'), (5, '大目标 AP'), (8, 'AR@100')]
        for ax, (idx, title) in zip(axes.flat, panels):
            lines(ax, lambda r, i=idx: r['validation']['coco_eval_bbox'][i] * 100, validation=True, init_index=idx)
            ax.axhline(reference['coco_eval_bbox'][idx] * 100, color='#666', ls=':', lw=1)
            style(ax, title + ' · val400', '指标（0–100）')
        lines(axes.flat[6], lambda r: r['train']['loss'])
        style(axes.flat[6], '总训练损失（增强 / 数据不同）', '加权损失')
        lines(axes.flat[7], lambda r: r['lr'][1])
        style(axes.flat[7], '检测头轮末学习率', '学习率')
        axes.flat[7].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        save(fig, 'overview', pdf)

        ann = json.loads((ROOT / 'data/annotations/val400.json').read_text())
        counts = {c['id']: sum(a['category_id'] == c['id'] for a in ann['annotations']) for c in ann['categories']}
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharey=True)
        fig.suptitle('各类别 mAP@50–95 · 虚线为共同起点权重的前一轮评测', fontsize=17)
        for ax, c in zip(axes.flat, ann['categories']):
            name = c['name']
            lines(ax, lambda r, n=name: r['validation']['per_class_ap'][n] * 100, validation=True)
            ax.axhline(reference['per_class_ap'][name] * 100, color='#666', ls=':', lw=1)
            ax.set_ylim(0, 100)
            style(ax, f'{name} · 验证 {counts[c["id"]]} 框', 'AP（0–100）')
        save(fig, 'class_ap', pdf)

        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle('训练损失分项与每轮耗时 · 全量训练无独立验证 AP', fontsize=17)
        for ax, family in zip(axes.flat, ['bbox', 'giou', 'vfl', 'fgl', 'ddf']):
            lines(ax, lambda r, f=family: sum(v for k, v in r['train'].items() if k.startswith('loss_' + f)))
            style(ax, 'loss_' + family, '各分支加权总和')
        lines(axes.flat[-1], lambda r: r['seconds'] / 60)
        style(axes.flat[-1], '每轮耗时（含验证 / 保存）', '分钟')
        save(fig, 'loss_components', pdf)
    (OUT / 'experiment_metrics.tmp.pdf').replace(OUT / 'experiment_metrics.pdf')
    summary = {'updated_at': stamp, 'sources': sources, 'starting_weight_evaluation_source': starting_source, 'runs': {}}
    flat = []
    for name, rows in data.items():
        v = [r for r in rows if r['validation'].get('coco_eval_bbox')]
        info = {'completed_epochs': len(rows), 'planned_epochs': RUNS[name][3], 'complete': (ROOT / 'runs' / name / 'COMPLETE').exists()}
        if v:
            best = max(v, key=lambda r: r['validation']['coco_eval_bbox'][0])
            info.update(latest_map=v[-1]['validation']['coco_eval_bbox'][0] * 100, best_map=best['validation']['coco_eval_bbox'][0] * 100, best_epoch=best['epoch'], latest_small_ap=v[-1]['validation']['coco_eval_bbox'][3] * 100)
        if name in initial:
            info['initial_map'] = initial[name][0] * 100
        summary['runs'][name] = info
        for r in rows:
            row = {'run': name, 'epoch': r['epoch'], 'seconds': r['seconds']}
            row.update({'train/' + k: v for k, v in r['train'].items()})
            row.update({'validation/' + k: v * 100 for k, v in zip(METRICS, r['validation'].get('coco_eval_bbox', []))})
            row.update({'class/' + k: v * 100 for k, v in r['validation'].get('per_class_ap', {}).items()})
            flat.append(row)
    with (OUT / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in flat for k in row)) or ['run', 'epoch'])
        writer.writeheader()
        writer.writerows(flat)
    temp = OUT / 'status.tmp.json'
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    temp.replace(OUT / 'status.json')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'plot.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.watch and (OUT / 'status.json').exists():
        time.sleep(max(0, 1200 - (time.time() - (OUT / 'status.json').stat().st_mtime)))
    while True:
        generate()
        if not args.watch or all((ROOT / 'runs' / n / 'COMPLETE').exists() for n in RUNS):
            break
        time.sleep(1200)
