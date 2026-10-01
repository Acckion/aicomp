"""Refresh query-level alignment experiment metrics once per minute."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/ir_query_alignment'
def read(path):
    try:return json.loads(path.read_text())
    except(FileNotFoundError,json.JSONDecodeError):return{}
def generate():
    OUT.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(2,3,figsize=(15,8))
    summary={}
    for job,label in [('ir_query800','Paired IR query'),('ir_query_control800','RGB control')]:
        directory=ROOT/'runs'/job;path=directory/'metrics.jsonl'
        rows=[json.loads(l)for l in path.read_bytes().splitlines(keepends=True)if l.endswith(b'\n')]if path.exists()else[]
        initial=read(directory/'initial_metrics.json')
        points=([(0,initial)]if initial else[])+[(r['epoch'],r['validation'])for r in rows]
        for ax,index,title in zip(axes.flat,[0,1,2,3],['mAP@50-95','AP50','AP75','AP small']):
            xy=[(epoch,metrics['coco_eval_bbox'][index]*100)for epoch,metrics in points]
            if xy:ax.plot(*zip(*xy),marker='.',label=label)
            ax.set_title(title);ax.set_xlabel('epoch');ax.set_xlim(0,8);ax.grid(alpha=.2)
            if ax.lines:ax.legend()
        xy=[(r['epoch'],r['validation']['ap_by_iou'].get('0.90',0)*100)for r in rows if'ap_by_iou'in r['validation']]
        if xy:axes[1,1].plot(*zip(*xy),marker='.',label=label)
        summary[job]={'completed_epochs':len(rows),'progress':read(directory/'optimizer_progress.json')}
    axes[1,1].set_title('AP90');axes[1,1].set_xlim(0,8);axes[1,1].grid(alpha=.2)
    if axes[1,1].lines:axes[1,1].legend()
    status=read(ROOT/'experiments/ir_query_alignment/status.json');summary['controller']=status
    axes[1,2].axis('off');axes[1,2].text(0,1,'GPU2 / per-process 8.5 GiB cap\n8 epochs paired + 8 epochs control\n\nStage: '+status.get('stage','preparing')+'\nJob: '+status.get('job','')+'\n\nTrain 1600 / validation 400\nPaired / shuffled / zero IR ablations\nOnly RGB-coordinate GT supervision',va='top')
    fig.suptitle('RGB-referenced query-level IR alignment');fig.tight_layout()
    tmp=OUT/'overview.tmp.png';fig.savefig(tmp,dpi=120);tmp.replace(OUT/'overview.png');plt.close(fig)
    summary['updated_at']=time.time();(OUT/'status.json').write_text(json.dumps(summary,indent=2))
    return status
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        status=generate()
        if not args.watch or status.get('stage')in['complete','failed']:break
        time.sleep(60)
