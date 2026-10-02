"""Minute-updated fresh pretraining comparison; complete epochs only."""
import fcntl
import json
import os
from pathlib import Path
import time
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring/scene_rgb'
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ['scene_rgb800', 'scene_rgb_fastcontrol800', 'scene_semantic_init800', 'scene_obj365_reset800', 'scene_obj365_pool800', 'scene_obj365_pool800_twowheel', 'scene_obj365_pool800_airqueries']


def evidence(all_rows):
    counts = json.loads((ROOT / 'experiments/scene_groups/report.json').read_text())['new_val_counts']
    classes = json.loads((ROOT / 'data/annotations/scene_val.json').read_text())['categories']
    name_counts = {c['name']:counts[str(c['id'])] for c in classes}
    summary = {'updated_at':time.strftime('%F %T'), 'validation':'scene-grouped 390; not phase2', 'runs':{}, 'paired':{},
               'limitations':'Warmup epochs1-3 are not final quality. Compare common epochs, not different-age peaks. Class tricycle has only5 GT. ETA assumes observed shared-server speed continues.'}
    for name, rows in all_rows.items():
        if not rows:
            summary['runs'][name] = {'completed_epochs':0}
            continue
        last = rows[-1]
        seconds = [r['seconds'] for r in rows[-3:] if 'seconds' in r]
        summary['runs'][name] = {'completed_epochs':last['epoch'],
            'map':100*last['validation']['coco_eval_bbox'][0],
            'small_ap':100*last['validation']['coco_eval_bbox'][3],
            'per_class_ap':{k:100*v for k,v in last['validation'].get('per_class_ap',{}).items()},
            'estimated_remaining_hours':statistics.median(seconds)*(100-last['epoch'])/3600 if seconds else None}
    for main, control in [('scene_semantic_init800','scene_rgb_fastcontrol800'),
                          ('scene_obj365_reset800','scene_rgb_fastcontrol800'),
                          ('scene_obj365_pool800','scene_obj365_reset800'),
                          ('scene_obj365_pool800_twowheel','scene_obj365_pool800'),
                          ('scene_obj365_pool800_airqueries','scene_obj365_pool800')]:
        a = {r['epoch']:r for r in all_rows[main]}; b = {r['epoch']:r for r in all_rows[control]}
        common = sorted(set(a)&set(b))
        # Three matched epochs after warmup; no automatic submission promotion.
        window = [e for e in common if e > 3][-3:]
        report = {'latest_common_epoch':common[-1] if common else None, 'window':window,
                  'status':'insufficient_post_warmup_evidence' if len(window)<3 else 'descriptive_matched_comparison'}
        if len(window)==3:
            for title,index in [('map',0),('ap75',2),('small_ap',3)]:
                report[title+'_mean_delta'] = statistics.mean(100*(a[e]['validation']['coco_eval_bbox'][index]-b[e]['validation']['coco_eval_bbox'][index]) for e in window)
            report['per_class_mean_delta'] = {name:{'delta':statistics.mean(100*(a[e]['validation']['per_class_ap'][name]-b[e]['validation']['per_class_ap'][name]) for e in window),
                                                     'gt_count':count,'sparse':count<20} for name,count in name_counts.items()}
        summary['paired'][main+' vs '+control] = report
    tmp = OUT / f'evidence.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(summary,indent=2))
    tmp.replace(OUT / 'evidence.json')
    return name_counts

def draw():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    all_rows = {}
    for name in RUNS:
        path = ROOT / 'runs' / name / 'metrics.jsonl'
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        all_rows[name] = rows
        for ax, index, title in zip(list(axes.flat)[:4], [0,1,2,3], ['mAP@50-95','AP50','AP75','Small AP']):
            pairs = [(r['epoch'],100*r['validation']['coco_eval_bbox'][index]) for r in rows if r.get('validation',{}).get('coco_eval_bbox')]
            if pairs:
                ax.plot(*zip(*pairs), marker='o', label=name)
            ax.set_title(title)
        pairs = [(r['epoch'],100*r['validation']['ap_by_iou']['0.90']) for r in rows if '0.90' in r.get('validation',{}).get('ap_by_iou',{})]
        if pairs:
            axes.flat[4].plot(*zip(*pairs), marker='o', label=name)
        pairs = [(r['epoch'],r['train']['loss']) for r in rows if 'loss' in r.get('train',{})]
        if pairs:
            axes.flat[5].plot(*zip(*pairs), marker='o', label=name)
    axes.flat[4].set_title('AP90')
    axes.flat[5].set_title('Loss')
    for ax in axes.flat:
        ax.set_xlabel('epoch')
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle('Fresh RGB: mapped class initialization vs reset | grouped holdout390 | not phase2 | '+time.strftime('%F %T'))
    fig.tight_layout()
    tmp = OUT / f'overview.{os.getpid()}.png'
    fig.savefig(tmp, dpi=130)
    plt.close(fig)
    tmp.replace(OUT / 'overview.png')
    counts = evidence(all_rows)
    fig, axes = plt.subplots(3,4,figsize=(16,10))
    for ax,(category,count) in zip(axes.flat, counts.items()):
        for name,rows in all_rows.items():
            pairs = [(r['epoch'],100*r['validation']['per_class_ap'][category]) for r in rows if category in r.get('validation',{}).get('per_class_ap',{})]
            if pairs:
                ax.plot(*zip(*pairs),marker='o',label=name)
        ax.set_title(category+' | GT='+str(count)+(' (sparse)' if count<20 else ''))
        ax.set_xlabel('epoch');ax.set_ylabel('AP@50-95');ax.grid(alpha=.25)
        ax.axvspan(0,3,color='gray',alpha=.08)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Per-class grouped holdout AP | gray=warmup | not phase2 | '+time.strftime('%F %T'))
    fig.tight_layout(rect=(0,.07,1,.96))
    tmp=OUT/f'per_class.{os.getpid()}.png';fig.savefig(tmp,dpi=130);plt.close(fig)
    tmp.replace(OUT/'per_class.png')
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    for method in ['pool','reset','coco_semantic']:
        for suffix in ['', '_b2']:
            stem='full2000_semantic_init800' if method=='coco_semantic' else 'full2000_obj365_'+method+'800'
            name=stem+suffix
            path=ROOT/'runs'/name/'metrics.jsonl'
            rows=[]
            if path.exists():
                for line in path.read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
            for ax,key,title in zip(list(axes.flat)[:5],['loss','loss_vfl','loss_bbox','loss_giou','loss_fgl'],['Loss','Classification loss','Box loss','GIoU loss','FGL loss']):
                pairs=[(r['epoch'],r['train'][key]) for r in rows if key in r.get('train',{})]
                if pairs:ax.plot(*zip(*pairs),marker='o',label=method+(' batch2' if suffix else ' batch1'))
                ax.set_title(title)
            pairs=[(r['epoch'],max(r['lr'])) for r in rows if r.get('lr')]
            if pairs:axes.flat[5].plot(*zip(*pairs),marker='o',label=method+(' batch2' if suffix else ' batch1'))
    axes.flat[5].set_title('Largest parameter-group LR')
    for ax in axes.flat:
        ax.set_xlabel('epoch');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=8)
    fig.suptitle('Full2000 public-pretraining candidates | NO independent AP | '+time.strftime('%F %T'))
    fig.tight_layout();tmp=OUT/f'full2000.{os.getpid()}.png';fig.savefig(tmp,dpi=130);plt.close(fig)
    tmp.replace(OUT/'full2000.png')

if __name__ == '__main__':
    lock = (OUT / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            draw()
        except Exception as error:
            print(repr(error), flush=True)
        time.sleep(60)
