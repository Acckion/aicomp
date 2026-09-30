"""Export fine-tune comparisons against the matching pre-fine-tune evaluations."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json,math,time,csv,fcntl,argparse,hashlib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_training import style,COLORS,METRICS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/finetune';OUT.mkdir(exist_ok=True)
RUNS={640:'ft_aug640',800:'ft_aug800'}

def snapshot():
    data={};initial={};sources={}
    for size,name in RUNS.items():
        path=ROOT/'runs'/name/'metrics.jsonl';raw=path.read_bytes()
        rows=[json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b'\n')]
        assert [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
        assert all(math.isfinite(v) for r in rows for v in r['train'].values())
        assert all(math.isfinite(v) for r in rows for v in r['validation']['coco_eval_bbox'])
        data[size]=rows
        initial[size]=json.loads((path.parent/'initial_metrics.json').read_text())['coco_eval_bbox']
        sources[str(size)]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'epochs':len(rows)}
    return data,initial,sources

def generate():
    data,initial,sources=snapshot();stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    baseline=[json.loads(s) for s in (ROOT/'runs/rgb1600/metrics.jsonl').read_text().splitlines()]
    best=max(baseline,key=lambda r:r['validation']['coco_eval_bbox'][0]);reference=best['validation']['coco_eval_bbox'][0]*100
    def finish(fig,name,pdf):
        fig.text(.02,.012,f'{stamp} | 独立 val400；本地指标不等同于 phase2 榜单。epoch 0=微调前；epoch 2 起为三卡训练。',fontsize=8,color='#555')
        fig.tight_layout(rect=(0,.045,1,.95))
        tmp=OUT/(name+'.tmp.png');fig.savefig(tmp,dpi=150);tmp.replace(OUT/(name+'.png'));pdf.savefig(fig);plt.close(fig)
    def curves(ax,fn,include_initial=None):
        for (size,rows),color in zip(data.items(),COLORS):
            x=[r['epoch'] for r in rows];y=[fn(r,size) for r in rows]
            if include_initial is not None:x=[0]+x;y=[include_initial(size)]+y
            ax.plot(x,y,color=color,lw=1.8,label=f'{size} 微调')
        ax.legend(fontsize=8)
    with PdfPages(OUT/'finetune_metrics.tmp.pdf') as pdf:
        fig,axes=plt.subplots(2,4,figsize=(18,9));fig.suptitle('D-FINE-X · 640 / 800 微调训练对照（计划 20 epochs）',fontsize=18)
        ax=axes[0,0];curves(ax,lambda r,s:r['validation']['coco_eval_bbox'][0]*100,lambda s:initial[s][0]*100)
        ax.axhline(reference,color='#555',ls='--',lw=1,label=f'原 640 baseline 最佳 {reference:.2f}')
        ax.legend(fontsize=7);style(ax,'总体 mAP@50–95','mAP（0–100）')
        ax=axes[0,1];curves(ax,lambda r,s:(r['validation']['coco_eval_bbox'][0]-initial[s][0])*100,lambda s:0)
        ax.axhline(0,color='#777',lw=1);style(ax,'相对同分辨率微调前的增益','mAP 变化（百分点）')
        for ax,idx,title in [(axes[0,2],3,'小目标 AP'),(axes[0,3],4,'中目标 AP'),(axes[1,0],5,'大目标 AP')]:
            curves(ax,lambda r,s,i=idx:r['validation']['coco_eval_bbox'][i]*100,lambda s,i=idx:initial[s][i]*100);style(ax,title,'AP50–95（0–100）')
        curves(axes[1,1],lambda r,s:r['train']['loss']);style(axes[1,1],'总训练损失','加权损失')
        curves(axes[1,2],lambda r,s:r['lr'][1]);style(axes[1,2],'检测头轮末学习率','学习率');axes[1,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        curves(axes[1,3],lambda r,s:r['seconds']/60);style(axes[1,3],'每轮耗时（含验证 / 保存）','分钟')
        finish(fig,'overview',pdf)
        ann=json.loads((ROOT/'data/annotations/val400.json').read_text());counts={c['id']:sum(a['category_id']==c['id'] for a in ann['annotations']) for c in ann['categories']}
        fig,axes=plt.subplots(3,4,figsize=(16,11),sharex=True,sharey=True);fig.suptitle('各类别 mAP@50–95 · 虚线为原 640 baseline 最佳轮次',fontsize=17)
        for ax,c in zip(axes.flat,ann['categories']):
            name=c['name'];curves(ax,lambda r,s,n=name:r['validation']['per_class_ap'][n]*100)
            ax.axhline(best['validation']['per_class_ap'][name]*100,color='#777',ls='--',lw=1)
            ax.set_ylim(0,100);style(ax,f'{name} · 验证 {counts[c["id"]]} 框','AP（0–100）')
        finish(fig,'class_ap',pdf)
        fig,axes=plt.subplots(2,3,figsize=(16,9));fig.suptitle('训练损失分项与验证召回',fontsize=17)
        for ax,family in zip(axes.flat,['bbox','giou','vfl','fgl','ddf']):
            curves(ax,lambda r,s,f=family:sum(v for k,v in r['train'].items() if k.startswith('loss_'+f)));style(ax,'loss_'+family,'各分支加权总和')
        curves(axes.flat[-1],lambda r,s:r['validation']['coco_eval_bbox'][8]*100);style(axes.flat[-1],'AR@100','AR（0–100）')
        finish(fig,'loss_components',pdf)
    (OUT/'finetune_metrics.tmp.pdf').replace(OUT/'finetune_metrics.pdf')
    summary={'updated_at':stamp,'sources':sources,'baseline_best_map':reference,'runs':{}}
    flat=[]
    for size,rows in data.items():
        scores=[initial[size][0]*100]+[r['validation']['coco_eval_bbox'][0]*100 for r in rows]
        i=max(range(len(scores)),key=scores.__getitem__)
        summary['runs'][str(size)]={'epoch':rows[-1]['epoch'],'latest_map':scores[-1],'initial_map':scores[0],'best_map':scores[i],'best_epoch':i,'best_gain_from_matching_initial':scores[i]-scores[0],'best_gain_from_original640':scores[i]-reference,'final_epoch20_map':scores[-1] if rows[-1]['epoch']==20 else None}
        for r in rows:
            d={'run':size,'epoch':r['epoch'],'seconds':r['seconds']};d.update({'train/'+k:v for k,v in r['train'].items()});d.update({'validation/'+k:v*100 for k,v in zip(METRICS,r['validation']['coco_eval_bbox'])});d.update({'class/'+k:v*100 for k,v in r['validation']['per_class_ap'].items()});flat.append(d)
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)));writer.writeheader();writer.writerows(flat)
    (OUT/'status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    lock=(OUT/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        generate()
        if not args.watch or all((ROOT/'runs'/n/'COMPLETE').exists() for n in RUNS.values()):break
        time.sleep(1200)
