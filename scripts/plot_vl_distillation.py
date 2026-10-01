"""Monitor both paired trials every minute; never invent AP for unfinished epochs."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt

from plot_training import style
from run_vl_distillation import ROOT,OUT as STATE,JOBS
from vl_distill_cache import write_json

OUT=ROOT/'monitoring/vl_distillation'
LABELS={'vl_control800':'无蒸馏对照','vl_distill800':'区域特征＋语义蒸馏'}


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def logged_progress(name):
    p=STATE/f'{name}.log'
    if not p.exists():return None
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-32768))
        tail=f.read().decode(errors='replace')
    matches=re.findall(r'Epoch: \[(\d+)/8\]\s+\[(\d+)/(\d+)\]',tail)
    if not matches:return None
    epoch,batch,total=map(int,matches[-1])
    return {'epoch':epoch+1,'batch':batch+1,'batches_per_epoch':total}


def rows(name):
    p=ROOT/'runs'/name/'metrics.jsonl'
    if not p.exists():return []
    result=[json.loads(line) for line in p.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
    assert [r['epoch'] for r in result]==list(range(1,len(result)+1))
    return result


def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    metric_panels=[(0,'mAP@50–95 · val400'),(2,'AP75 · val400'),('ap90','AP90 · val400'),(3,'小目标AP · val400')]
    data={name:rows(name) for name in JOBS}
    for ax,(index,title) in zip(axes.flat,metric_panels):
        for (name,label),color in zip(LABELS.items(),['#73814d','#2766a2']):
            values=[];initial=read(ROOT/'runs'/name/'initial_metrics.json')
            if initial:
                v=initial.get('ap_by_iou',{}).get('0.90') if index=='ap90' else initial['coco_eval_bbox'][index]
                if v is not None:values.append((0,v*100))
            for r in data[name]:
                val=r['validation'];v=val.get('ap_by_iou',{}).get('0.90') if index=='ap90' else val['coco_eval_bbox'][index]
                if v is not None:values.append((r['epoch'],v*100))
            if values:ax.plot(*zip(*values),marker='.',label=label,color=color)
        style(ax,title,'AP（0–100）');ax.set_xlim(0,8)
        if ax.lines:ax.legend(fontsize=9)
    for name,label in LABELS.items():
        r=data[name];x=[a['epoch'] for a in r]
        if r:
            axes[1,1].plot(x,[a['train'].get('loss_vl_feature',0) for a in r],label=label+'：特征')
            axes[1,1].plot(x,[a['train'].get('loss_vl_semantic',0) for a in r],label=label+'：语义',ls='--')
    style(axes[1,1],'额外蒸馏损失（加权）','损失')
    axes[1,1].set_xlim(0,8)
    if axes[1,1].lines:axes[1,1].legend(fontsize=8)
    status={'updated_at':datetime.now().astimezone().isoformat(),'teacher':read(STATE/'teacher/status.json'),'runs':{}}
    lines=[]
    for name in JOBS:
        controller=read(STATE/f'{name}_status.json');progress=read(ROOT/'runs'/name/'optimizer_progress.json')
        live=logged_progress(name)
        complete=(ROOT/'runs'/name/'COMPLETE').exists()
        status['runs'][name]={'controller':controller,'progress':progress,'logged_progress':live,'epochs':len(data[name]),'complete':complete}
        location=f"epoch {live['epoch']} / batch {live['batch']}/{live['batches_per_epoch']}" if live else f"epoch {progress.get('epoch','—')} / batch {progress.get('next_batch','—')}"
        lines.append(f"{LABELS[name]} · GPU{JOBS[name]}\n{len(data[name])}/8轮 · {controller.get('stage','准备中')}\n"
                     +location)
    teacher=status['teacher'];lines.append(f"训练目标特征缓存：{teacher.get('completed',0)}/{teacher.get('total',11387)}")
    axes[1,2].axis('off');axes[1,2].text(0,1,'\n\n'.join(lines),va='top',fontsize=10)
    fig.suptitle('D-FINE-X · 训练期视觉语言知识蒸馏对照',fontsize=17)
    fig.text(.02,.015,'每60秒更新｜同父权重、训练1600/验证400｜仅完成epoch产生AP｜训练缓存不含val400｜推理只使用D-FINE',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    tmp=OUT/'overview.tmp.png';fig.savefig(tmp,dpi=125);tmp.replace(OUT/'overview.png');plt.close(fig)
    write_json(OUT/'status.json',status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'plot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=generate()
        if not args.watch or all(r['complete'] or r['controller'].get('stage')=='failed' for r in status['runs'].values()):break
        time.sleep(60)
