"""Update completed-epoch metrics and live progress once per minute."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from plot_training import style

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/instance_memory'
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def generate():
    OUT.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(2,3,figsize=(15,8))
    for name,label,color in [('instance_memory800','实例记忆','#396d98'),('instance_memory_control800','禁用记忆同预算对照','#9a6847')]:
        run=ROOT/'runs'/name;p=run/'metrics.jsonl'
        rows=[json.loads(line)for line in p.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')] if p.exists() else []
        initial=read(run/'initial_metrics.json')
        for ax,index,title in zip(axes.flat,[0,1,2,3],['mAP@50–95','AP50','AP75','小目标 AP']):
            points=([(0,initial['coco_eval_bbox'][index]*100)] if initial else [])
            points+=[(r['epoch'],r['validation']['coco_eval_bbox'][index]*100)for r in rows]
            if points:ax.plot(*zip(*points),marker='.',color=color,label=label)
            style(ax,title,'AP（0–100）');ax.set_xlim(0,8)
        points=[(r['epoch'],r['validation']['ap_by_iou']['0.90']*100)for r in rows]
        if points:axes[1,1].plot(*zip(*points),marker='.',color=color,label=label)
    style(axes[1,1],'AP90','AP（0–100）');axes[1,1].set_xlim(0,8)
    state=read(ROOT/'experiments/instance_memory/status.json');progress=read(ROOT/'runs'/state.get('run','instance_memory800')/'optimizer_progress.json')
    axes[1,2].axis('off');axes[1,2].text(0,1,f"{state.get('stage','准备中')} · GPU1\n{state.get('run','—')}\n\n8轮记忆＋8轮同预算对照\n固定训练1600/验证400\n记忆来自训练GT，不含val/test\n\nepoch {progress.get('epoch','—')} · batch {progress.get('next_batch','—')}\n每卡显存硬上限8.5GiB",va='top',fontsize=11)
    for ax in axes.flat:
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('D-FINE-X · 训练实例记忆与定位适配',fontsize=17);fig.tight_layout(rect=(0,.02,1,.95))
    temp=OUT/'overview.tmp.png';fig.savefig(temp,dpi=120);temp.replace(OUT/'overview.png');plt.close(fig)
    (OUT/'status.json').write_text(json.dumps({'time':time.time(),'controller':state,'progress':progress},indent=2));return state
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        state=generate()
        if not args.watch or state.get('stage') in ['failed','complete']:break
        time.sleep(60)
