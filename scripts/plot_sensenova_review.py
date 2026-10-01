"""Refresh the frozen-candidate review monitor every minute."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/sensenova_review'
MONITOR=ROOT/'monitoring/sensenova_review'

def read(name):
    try:return json.loads((OUT/name).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def generate():
    MONITOR.mkdir(parents=True,exist_ok=True)
    worker,controller,summary=read('status.json'),read('controller_status.json'),read('summary.json')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    ax=axes[0,0];done=worker.get('completed_requests',0);total=max(worker.get('total_requests',864),1)
    ax.barh(['Region reviews'],[100*done/total],color='#36749d');ax.set_xlim(0,100)
    ax.set_xlabel('% complete');ax.set_title(f'{done}/{total} requests; boxes and categories fixed')
    ax=axes[0,1];ax.axis('off')
    lines=[f"Controller: {controller.get('stage','preparing')}; worker: {worker.get('stage','preparing')}",
           f"Image / proposal / variant: {worker.get('image_id','-')} / {worker.get('box_index','-')} / {worker.get('variant','-')}",
           f"Paired regions: {summary.get('paired_regions',0)}; fully reviewed images: {summary.get('paired_images',0)}/24",
           f"GPU7 physical cards: {controller.get('gpus',[1,5,6,7])}; 8.5 GiB PyTorch cap each",
           'Blind 12-class recognition + background + unknown; no detector label in prompt',
           'Low option mass / tiny / unknown: abstain; one-way penalties only',
           'Fixed 12 proposals/image; other scores unchanged; no phase2 or fitting']
    error=worker.get('error') or controller.get('error')
    if error:lines.append('ERROR: '+error[:130])
    ax.text(0,1,'\n\n'.join(lines),va='top',fontsize=9)
    ax=axes[1,0];diag=summary.get('diagnostics',{});names=list(diag)
    if names:
        x=list(range(len(names)));width=.25
        for off,key,label in [(-width,'native_score_auc','D-FINE native'),(0,'vlm_support_auc','VLM support'),(width,'penalty_half_score_auc','After penalty')]:
            values=[diag[n].get(key) for n in names]
            ax.bar([i+off for i in x],[v if v is not None else 0 for v in values],width,label=label)
        ax.set_xticks(x,names);ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Waiting for paired region reviews',ha='center',transform=ax.transAxes)
    ax.set_ylim(0,1);ax.set_title('TP/FP discrimination AUC: diagnostic, not detection AP')
    ax=axes[1,1];ev=summary.get('evaluations',{})
    if ev:
        ref=ev['dfine_original']['map50_95'];names=[n for n in ev if n!='dfine_original']
        ax.bar(range(len(names)),[ev[n]['map50_95']-ref for n in names],color='#547da1')
        ax.set_xticks(range(len(names)),names,rotation=40,ha='right',fontsize=7)
        ax.axhline(0,color='black',lw=.7)
        ax.set_ylabel('mAP points vs D-FINE');ax.set_title(f'Partial-review mAP@50-95 delta; baseline {ref:.2f}')
    else:ax.text(.5,.5,'Waiting for first fully reviewed image\nNo mAP conclusion yet',ha='center',transform=ax.transAxes)
    fig.suptitle('SenseNova as D-FINE candidate-region reviewer',fontsize=15)
    fig.text(.01,.01,datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')+' | Refresh 60s | Small validation pilot; not phase2 performance',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94));temp=MONITOR/'overview.tmp.png';fig.savefig(temp,dpi=120);temp.replace(MONITOR/'overview.png');plt.close(fig)
    return controller.get('stage') in ['complete','failed','stopped_by_user']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        terminal=generate()
        if terminal or not args.watch:break
        time.sleep(60)
