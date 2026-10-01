"""Refresh auxiliary/control validation and positive-assignment curves every minute."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from plot_training import style

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'monitoring/aux_detection'
def rows(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):continue
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result
def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    counts={}
    for name,label,color in [('aux_detection800','训练专用密集头','#b55a3c'),
                             ('aux_detection_control800','同预算对照','#486b8a')]:
        records=rows(ROOT/'runs'/name/'metrics.jsonl');counts[name]=len(records)
        for axis,index,title in zip(axes.flat,[0,2,3],['mAP@50–95','AP75','小目标 AP']):
            points=[(r['epoch'],r['validation']['coco_eval_bbox'][index]*100)for r in records]
            if points:axis.plot(*zip(*points),marker='.',label=label,color=color)
            style(axis,title,'AP（0–100）')
        points=[(r['epoch'],r['validation']['ap_by_iou']['0.90']*100)for r in records]
        if points:axes[1,0].plot(*zip(*points),marker='.',label=label,color=color)
    assignment=rows(ROOT/'runs/aux_detection800/aux_assignment.jsonl')
    if assignment:
        axes[1,1].plot(range(len(assignment)),[r['positives_per_gt']for r in assignment],label='每 GT 正位置')
        axes[1,1].plot(range(len(assignment)),[r['zero_positive_gt']for r in assignment],label='零正位置 GT')
    style(axes[1,0],'AP90','AP（0–100）');style(axes[1,1],'ATSS 派生分配统计','数量')
    try:controller=json.loads((ROOT/'experiments/aux_detection/status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):controller={}
    axes[1,2].axis('off')
    axes[1,2].text(0,1,f"GPU7 · {controller.get('stage','准备中')}\n\n"
                 f"辅助头 {counts['aux_detection800']}/12轮\n对照 {counts['aux_detection_control800']}/12轮\n\n"
                 '训练1600 / 验证400\n有效 batch12 · BN统计固定\n原生单模型 Top100 推理',va='top')
    for axis in axes.flat[:5]:
        if axis.lines:axis.legend(fontsize=9)
    fig.suptitle('D-FINE-X · 真实一对多辅助检测监督',fontsize=16)
    fig.text(.02,.01,'每60秒更新｜辅助头仅训练执行，推理删除｜不是完整Co-DETR｜没有phase2成绩',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));temp=OUT/'overview.tmp.png'
    fig.savefig(temp,dpi=120);temp.replace(OUT/'overview.png');plt.close(fig)
    (OUT/'status.json').write_text(json.dumps({'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                                              'epochs':counts,'controller':controller},indent=2))
    return controller
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        state=generate()
        if not args.watch or state.get('stage')in['failed','complete']:break
        time.sleep(60)
