"""Minute monitoring for full-frame native pixels versus low-detail sham."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'monitoring/whole_frame_head'
NAMES=['scene_whole_head_real','scene_whole_head_sham',
       'scene_whole_encoder_real','scene_whole_encoder_sham']


def read(name):
    p=ROOT/'runs'/name/'metrics.jsonl';rows=[]
    if p.exists():
        for line in p.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):continue
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    initial=ROOT/'runs'/name/'initial_metrics.json'
    if initial.exists():
        data=json.loads(initial.read_text());validation=data.get('validation',data)
        if 'coco_eval_bbox' in validation:rows.insert(0,{'epoch':0,'validation':validation})
    return rows


def draw():
    rows={n:read(n) for n in NAMES};fig,axes=plt.subplots(2,3,figsize=(14,8))
    evidence={'updated_at':time.strftime('%F %T'),'geometry':[1088,1920],
        'validation':'grouped390; not phase2','runs':{},'paired':None,
        'limitations':'Head arms freeze backbone and encoder; encoder arms train encoder plus decoder. Backbone and BN statistics stay frozen. Same geometry and schedule. Epoch1 is warmup. Need at least3 matched post-warmup epochs; do not compare different-age peaks.'}
    for name,data in rows.items():
        label=('Encoder+head' if '_encoder_' in name else 'Head only')+' / '+('original' if name.endswith('real') else 'reconstructed800')
        for ax,index,title in zip(list(axes.flat)[:4],[0,1,2,3],['mAP@50-95','AP50','AP75','Small AP']):
            values=[(r['epoch'],100*r['validation']['coco_eval_bbox'][index]) for r in data if 'coco_eval_bbox' in r.get('validation',{})]
            if values:ax.plot(*zip(*values),marker='o',label=label)
            ax.set_title(title)
        values=[(r['epoch'],100*r['validation']['ap_by_iou']['0.90']) for r in data if '0.90' in r.get('validation',{}).get('ap_by_iou',{})]
        if values:axes.flat[4].plot(*zip(*values),marker='o',label=label)
        values=[(r['epoch'],r['train']['loss']) for r in data if 'loss' in r.get('train',{})]
        if values:axes.flat[5].plot(*zip(*values),marker='o',label=label)
        evidence['runs'][name]={'completed_epochs':max([r['epoch'] for r in data],default=0),
            'map':data[-1]['validation']['coco_eval_bbox'][0]*100 if data else None}
    axes.flat[4].set_title('AP90');axes.flat[5].set_title('Loss')
    for ax in axes.flat:
        ax.set_xlabel('continuation epoch');ax.grid(alpha=.25);ax.axvspan(0,1,color='gray',alpha=.08)
        if ax.lines:ax.legend(fontsize=9)
    evidence['paired_by_comparison']={}
    for key,left,right in [('head_real_vs_sham',NAMES[0],NAMES[1]),
                           ('encoder_real_vs_sham',NAMES[2],NAMES[3]),
                           ('encoder_vs_head_real',NAMES[2],NAMES[0]),
                           ('encoder_vs_head_sham',NAMES[3],NAMES[1])]:
        a={r['epoch']:r for r in rows[left]};b={r['epoch']:r for r in rows[right]}
        common=[e for e in sorted(a.keys()&b.keys()) if e>=2][-3:]
        if len(common)==3:
            comparison={'epochs':common,**{metric:statistics.mean(100*(a[e]['validation']['coco_eval_bbox'][index]-b[e]['validation']['coco_eval_bbox'][index]) for e in common)
                for metric,index in [('map_delta',0),('AP75_delta',2),('small_AP_delta',3)]}}
            evidence['paired_by_comparison'][key]=comparison
            if key=='head_real_vs_sham':evidence['paired']=comparison
    fig.suptitle('Full-frame adaptation | gray=warmup | not phase2 | '+evidence['updated_at']);fig.tight_layout()
    temp=OUT/f'overview.{os.getpid()}.png';fig.savefig(temp,dpi=130);plt.close(fig);temp.replace(OUT/'overview.png')
    temp=OUT/f'evidence.{os.getpid()}.tmp';temp.write_text(json.dumps(evidence,indent=2)+'\n');temp.replace(OUT/'evidence.json')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:draw()
        except Exception as error:print(repr(error),flush=True)
        time.sleep(60)
