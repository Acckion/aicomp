"""Refresh the four overnight experiments and shared GPU state every minute."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import subprocess
import time
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_training import style

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'monitoring/breakthrough_night'
RUNS={
    'ir_query800':('IR目标邻域','experiments/ir_query_alignment/status.json'),
    'aux_detection800':('一对多检测头','experiments/aux_detection/status.json'),
    'codino1600':('Co-DINO迁移','experiments/codino/status.json'),
    'instance_memory800':('训练实例记忆','experiments/instance_memory/status.json'),
}
CONTROLS={'ir_query800':'ir_query_control800','aux_detection800':'aux_detection_control800',
          'instance_memory800':'instance_memory_control800'}

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def rows(run):
    p=ROOT/'runs'/run/'metrics.jsonl';result=[]
    if not p.exists():return result
    for line in p.read_bytes().splitlines(keepends=True):
        if line.endswith(b'\n'):
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    return result

def points(run,metric):
    result=[]
    initial=read(ROOT/'runs'/run/'initial_metrics.json')
    candidates=([(0,initial)]if initial else [])+[(r.get('epoch',0),r.get('validation',{}))for r in rows(run)]
    for epoch,v in candidates:
        value=v.get('ap_by_iou',{}).get('0.90') if metric=='ap90' else (v.get('coco_eval_bbox',[])[metric]if len(v.get('coco_eval_bbox',[]))>metric else None)
        if value is not None:result.append((epoch,value*100))
    return result

def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    statuses={run:read(ROOT/path)for run,(_,path)in RUNS.items()}
    gpu=[]
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        for line in output.splitlines():
            index,used,free,util=[int(x.strip())for x in line.split(',')]
            gpu.append({'index':index,'used_mib':used,'free_mib':free,'utilization_percent':util})
    except (subprocess.SubprocessError,ValueError):pass
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,metric,title in zip(axes.flat,[0,2,3,'ap90'],['mAP@50–95 · RGB输出','AP75 · RGB输出','小目标 AP · RGB输出','AP90 · RGB输出']):
        for number,(run,(label,_))in enumerate(RUNS.items()):
            color=plt.get_cmap('tab10')(number)
            data=points(run,metric)
            if data:ax.plot(*zip(*data),marker='.',label=label,color=color)
            control=points(CONTROLS[run],metric) if run in CONTROLS else []
            if control:ax.plot(*zip(*control),linestyle='--',label=label+'对照',color=color,alpha=.7)
        style(ax,title,'AP（0–100）')
        if ax.lines:
            ax.legend(fontsize=8)
            last=max(max(line.get_xdata())for line in ax.lines)
            ax.set_xlim(0,max(1,last));ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if last==0:
                values=[float(y)for line in ax.lines for y in line.get_ydata()]
                center=(min(values)+max(values))/2;ax.set_ylim(center-2.5,center+2.5)
    ir=points('ir_detector800',0)
    if ir:axes[1,1].plot(*zip(*ir),color='#9c5534',marker='.')
    style(axes[1,1],'IR单模态诊断 · 单独展示','mAP@50–95')
    axes[1,2].axis('off');lines=[]
    for run,(label,_)in RUNS.items():
        s=statuses[run]
        if run=='instance_memory800'and not s:s=read(ROOT/'experiments/instance_memory/cache_status.json')
        lines.append(f"{label}：{s.get('stage','准备中')} · {len(rows(run))}轮")
    lines.extend(['','共享显卡总占用 / 利用率（含他人任务）'])
    lines.extend([f"GPU{x['index']}：{x['used_mib']/1024:.1f}GiB / {x['utilization_percent']}%"for x in gpu])
    axes[1,2].text(0,1,'\n'.join(lines),va='top',fontsize=10)
    updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    fig.suptitle('AICOMP · 四方向夜间实验',fontsize=18)
    fig.text(.02,.015,f'60秒刷新｜独立val400｜本地AP不代表phase2成绩｜{updated}',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    tmp=OUT/'overview.tmp.png';fig.savefig(tmp,dpi=120);tmp.replace(OUT/'overview.png');plt.close(fig)
    summary={'updated_at':updated,'experiments':statuses,'gpu_shared_totals':gpu}
    (OUT/'status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        generate()
        if not args.watch:break
        time.sleep(60)
