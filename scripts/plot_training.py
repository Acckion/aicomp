#!/usr/bin/env python3
"""Plot completed training epochs; --watch updates artifacts on log changes."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse, csv, json, math, time, hashlib, fcntl
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring'
OUT.mkdir(exist_ok=True)
font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120})
NAMES = {'rgb1600': '1600 训练 + 400 验证', 'rgb2000': '2000 全量训练'}
COLORS = ['#2F5D9B', '#C47B32']
METRICS = ['mAP@50–95','AP50','AP75','AP-small','AP-medium','AP-large','AR@1','AR@10','AR@100','AR-small','AR-medium','AR-large']

def read():
    result, receipts = {}, {}
    for name in NAMES:
        path = ROOT / 'runs' / name / 'metrics.jsonl'
        raw = path.read_bytes()
        rows = []
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b'\n'): break
            rows.append(json.loads(line))
        epochs = [r['epoch'] for r in rows]
        if epochs != list(range(1, len(rows)+1)):
            raise ValueError(f'{name}: duplicate or missing epochs: {epochs}')
        for row in rows:
            for key, value in row['train'].items():
                if not math.isfinite(value): raise ValueError(f'{name}: nonfinite {key}')
            total = sum(v for k,v in row['train'].items() if k.startswith('loss_'))
            if not math.isclose(total, row['train']['loss'], rel_tol=1e-5):
                raise ValueError(f'{name}: loss component sum mismatch')
        result[name] = rows
        receipts[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'completed_epochs': len(rows)}
    return result, receipts

def style(ax, title, ylabel=''):
    ax.set_title(title, loc='left', fontsize=12)
    ax.set_xlabel('Epoch（已完成轮次）')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.xaxis.get_major_locator().set_params(integer=True)

def save(fig, name, stamp, pdf=None):
    fig.text(.02, .012, f'来源：runs/*/metrics.jsonl | {stamp} | 仅完整轮次；未平滑。全量版无独立验证集。', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0,.045,1,.95))
    tmp = OUT / (name+'.tmp.png')
    fig.savefig(tmp, dpi=150, facecolor='white')
    tmp.replace(OUT / (name+'.png'))
    if pdf: pdf.savefig(fig)
    plt.close(fig)

def generate():
    data, receipts = read()
    stamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    val = data['rgb1600']
    if not val or not data['rgb2000']: return
    vx = [r['epoch'] for r in val]
    def compare(ax, fn):
        for (name, rows),color in zip(data.items(),COLORS):
            ax.plot([r['epoch'] for r in rows], [fn(r) for r in rows], color=color, label=NAMES[name], lw=1.7)
        ax.legend(fontsize=8)
    def validation(ax, indices):
        for i in indices:
            ax.plot(vx, [r['validation']['coco_eval_bbox'][i]*100 for r in val], label=METRICS[i], lw=1.6)
        ax.set_ylim(0,100); ax.legend(fontsize=8)
    with PdfPages(OUT / 'training_metrics.tmp.pdf') as pdf:
        fig, axs = plt.subplots(2,3,figsize=(16,9))
        fig.suptitle('D-FINE-X RGB baseline · 训练监控（计划 100 epochs）', fontsize=18)
        compare(axs[0,0], lambda r:r['train']['loss']); style(axs[0,0],'总训练损失','加权损失')
        validation(axs[0,1],[0,1,2]); style(axs[0,1],'400 张独立验证集 · 检测 AP','AP（0–100）')
        validation(axs[0,2],[3,4,5]); style(axs[0,2],'不同目标尺寸 · AP50:95','AP（0–100）')
        validation(axs[1,0],[6,7,8]); style(axs[1,0],'验证集 · 平均召回率','AR（0–100）')
        for i,label in [(1,'检测头'),(0,'骨干网络')]:
            axs[1,1].plot(vx,[r['lr'][i] for r in val],label=label)
        axs[1,1].set_yscale('log'); axs[1,1].legend(); style(axs[1,1],'动态学习率（1600 版轮末值）','学习率 · 对数刻度')
        compare(axs[1,2],lambda r:r['seconds']); style(axs[1,2],'每轮总耗时（含验证 / 保存）','秒')
        save(fig,'overview',stamp,pdf)
        maps = [r['validation']['coco_eval_bbox'][0]*100 for r in val]
        best_index = max(range(len(maps)), key=maps.__getitem__)
        final_text = f'{maps[-1]:.2f}' if vx[-1] == 100 else '待第 100 轮完成'
        fig, ax = plt.subplots(figsize=(12,6))
        fig.suptitle('总体 mAP@50–95 · 400 张独立验证集', fontsize=18)
        ax.plot(vx, maps, color=COLORS[0], lw=2, label='mAP@50–95（全部 12 类平均）')
        ax.scatter([vx[best_index]], [maps[best_index]], color=COLORS[1], s=65, zorder=5, label='历史最佳')
        ax.set_ylim(0,100)
        style(ax, f'最新：{maps[-1]:.2f}（epoch {vx[-1]}）  |  最佳：{maps[best_index]:.2f}（epoch {vx[best_index]}）\n最终（epoch 100）：{final_text}', 'mAP@50–95（0–100）')
        ax.legend(loc='lower right')
        save(fig,'map50_95',stamp,pdf)
        counts = Counter(a['category_id'] for a in json.loads((ROOT/'data/annotations/val400.json').read_text())['annotations'])
        cats = json.loads((ROOT/'data/annotations/val400.json').read_text())['categories']
        fig,axs = plt.subplots(3,4,figsize=(16,11),sharex=True,sharey=True)
        fig.suptitle('各类别验证 AP50:95 · 小样本类别波动需谨慎解读',fontsize=18)
        for ax, cat in zip(axs.flat,cats):
            name=cat['name']; ys=[r['validation']['per_class_ap'][name]*100 for r in val]
            ax.plot(vx,ys,color=COLORS[0]); ax.set_ylim(0,100)
            style(ax,f'{name} · {counts[cat["id"]]} 个标注框','AP（0–100）')
            ax.text(.97,.08,f'最新 {ys[-1]:.2f}',transform=ax.transAxes,ha='right',fontsize=9)
        save(fig,'class_ap',stamp,pdf)
        fig,axs=plt.subplots(2,3,figsize=(16,9))
        fig.suptitle('加权损失分项 · 包含各辅助 / 去噪分支的同类项总和',fontsize=17)
        for ax,family in zip(axs.flat,['bbox','giou','vfl','fgl','ddf']):
            compare(ax,lambda r,f=family:sum(v for k,v in r['train'].items() if k.startswith('loss_'+f)))
            style(ax,'loss_'+family,'加权损失')
        validation(axs.flat[-1],[9,10,11]); style(axs.flat[-1],'不同尺寸 · 平均召回率','AR（0–100）')
        save(fig,'loss_components',stamp,pdf)
        keys=sorted(k for k in val[0]['train'] if k.startswith('loss_'))
        for start in range(0,len(keys),12):
            fig,axs=plt.subplots(3,4,figsize=(16,11))
            fig.suptitle('完整训练损失明细 · 原始日志加权值',fontsize=17)
            for ax,key in zip(axs.flat,keys[start:start+12]):
                compare(ax,lambda r,k=key:r['train'][k]); style(ax,key)
            for ax in list(axs.flat)[len(keys[start:start+12]):]: ax.set_visible(False)
            fig.tight_layout(rect=(0,0,1,.95)); pdf.savefig(fig); plt.close(fig)
    (OUT/'training_metrics.tmp.pdf').replace(OUT/'training_metrics.pdf')
    flat=[]
    for name,rows in data.items():
        for r in rows:
            d={'run':name,'epoch':r['epoch'],'seconds':r['seconds']}
            d.update({'train/'+k:v for k,v in r['train'].items()})
            d.update({'lr/group_'+str(i):v for i,v in enumerate(r['lr'])})
            d.update({'val/'+k:100*v for k,v in zip(METRICS,r['validation'].get('coco_eval_bbox',[]))})
            d.update({'val/class/'+k:100*v for k,v in r['validation'].get('per_class_ap',{}).items()})
            flat.append(d)
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for d in flat for k in d))); writer.writeheader(); writer.writerows(flat)
    summary={'updated_at':stamp,'sources':receipts,'checks':'completed epochs consecutive; training values finite; loss components sum verified','runs':{name:{'epoch':rows[-1]['epoch'],'loss':rows[-1]['train']['loss'],'best_ap_points':None if rows[-1]['best_ap'] is None else rows[-1]['best_ap']*100,'best_epoch':rows[-1]['best_epoch']} for name,rows in data.items()}}
    summary['map50_95'] = {
        'dataset': 'val400', 'scale': '0–100',
        'latest': maps[-1], 'latest_epoch': vx[-1],
        'best': maps[best_index], 'best_epoch': vx[best_index],
        'final_epoch_100': maps[-1] if vx[-1] == 100 else None,
        'rgb2000': None, 'rgb2000_note': '无独立验证集'
    }
    (OUT/'status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--watch',action='store_true'); args=parser.parse_args()
    lock=(OUT/'plot.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last=None
    while True:
        sig=tuple((ROOT/'runs'/n/'metrics.jsonl').stat().st_mtime_ns for n in NAMES)
        if sig!=last:
            generate(); last=sig
        if not args.watch or all((ROOT/'runs'/n/'COMPLETE').exists() for n in NAMES): break
        time.sleep(1200)
