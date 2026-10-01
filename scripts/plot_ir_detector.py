"""IR-only training and actual complementary GT coverage, refreshed every minute."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from plot_training import style

ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'experiments/multimodal_probe'
OUT=ROOT/'monitoring/ir_detector'
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'runs/ir_detector800/metrics.jsonl'
    rows=[json.loads(l)for l in p.read_bytes().splitlines(keepends=True) if l.endswith(b'\n')] if p.exists() else []
    initial=read(ROOT/'runs/ir_detector800/initial_metrics.json')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,index,title in zip(axes.flat,[0,1,2,3],['IR mAP@50–95','IR AP50','IR AP75','IR 小目标 AP']):
        points=([(0,initial['coco_eval_bbox'][index]*100)] if initial else [])
        points+=[(r['epoch'],r['validation']['coco_eval_bbox'][index]*100)for r in rows]
        if points:ax.plot(*zip(*points),marker='.',color='#ad593f')
        style(ax,title,'AP（0–100）');ax.set_xlim(0,24)
    reports={e:read(STATE/f'ir_epoch_{e:03d}/report.json')for e in [3,6,12,24]}
    for threshold,label in [('0.5','IoU≥0.50'),('0.75','IoU≥0.75'),('0.9','IoU≥0.90')]:
        points=[(e,r['coverage'][threshold]['ir_unique'])for e,r in reports.items() if r]
        if points:axes[1,1].plot(*zip(*points),marker='o',label=label)
    style(axes[1,1],'IR 独有覆盖GT · 诊断','目标数量');axes[1,1].set_xlim(0,24)
    if axes[1,1].lines:axes[1,1].legend(fontsize=9)
    controller=read(STATE/'ir_detector_status.json');progress=read(ROOT/'runs/ir_detector800/optimizer_progress.json')
    summary={'updated_at':datetime.now().astimezone().isoformat(),'epochs':len(rows),'controller':controller,
             'progress':progress,'complementary_evaluations_completed':[e for e,r in reports.items() if r]}
    axes[1,2].axis('off');axes[1,2].text(0,1,
        f"{len(rows)}/24 epoch · {controller.get('stage','准备中')}\n\n"
        f"GPU5、6训练 · 每卡8.5GiB上限\n有效batch12 · 训练1600/验证400\n\n"
        f"进度：epoch {progress.get('epoch','—')} / batch {progress.get('next_batch','—')}\n"
        f"互补诊断已完成：{summary['complementary_evaluations_completed']}",va='top',fontsize=11)
    fig.suptitle('D-FINE-X · 红外检测适配与RGB互补诊断',fontsize=17)
    fig.text(.02,.01,'每60秒更新｜独立验证400｜IR独有GT覆盖不等于可部署融合收益｜没有phase2成绩',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    tmp=OUT/'overview.tmp.png';fig.savefig(tmp,dpi=120);tmp.replace(OUT/'overview.png');plt.close(fig)
    (OUT/'status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        result=generate()
        if not args.watch or result['controller'].get('stage')=='failed':break
        if result['controller'].get('stage')=='complete' and len(result['complementary_evaluations_completed'])==4:break
        time.sleep(60)
