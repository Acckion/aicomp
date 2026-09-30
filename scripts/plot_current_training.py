"""Show current detail experiment honestly, including interruption and full-data progress."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json,re,time,argparse,fcntl
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_training import style
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/current_training'


def rows(name):
 p=ROOT/'runs'/name/'metrics.jsonl'
 return [json.loads(x) for x in p.read_bytes().splitlines(keepends=True) if x.endswith(b'\n')] if p.exists() else []


def generate(cached=False):
 OUT.mkdir(parents=True,exist_ok=True)
 if cached and (OUT/'snapshot.json').exists():
  previous=json.loads((OUT/'snapshot.json').read_text());full,detail=previous['full'],previous['detail']
 else:full,detail=rows('ft2000_aug800'),rows('detail800')
 initial_path=ROOT/'runs/detail800/initial_metrics.json'
 initial=json.loads(initial_path.read_text())['coco_eval_bbox'] if initial_path.exists() else None
 state=json.loads((ROOT/'experiments/after_targetcrop/status.json').read_text())
 current_log=ROOT/'experiments/after_targetcrop/detail800.log'
 text=current_log.read_text(errors='replace') if current_log.exists() else ''
 step=re.findall(r'Epoch: \[(\d+)/20\]\s+\[\s*(\d+)/\d+\].*?\bloss: ([\d.]+) \(([\d.]+)\)',text)
 stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
 fig,axes=plt.subplots(2,3,figsize=(16,9));fig.suptitle(f'D-FINE-X · 当前训练监控\n细节分支 {len(detail)}/20 · 全量微调 {len(full)}/20 · 队列状态：{state["stage"]}',fontsize=17)
 for ax,idx,title in [(axes[0,0],0,'细节分支 mAP@50–95 · val400'),(axes[0,1],3,'细节分支小目标 AP · val400')]:
  x=[r['epoch'] for r in detail];y=[r['validation']['coco_eval_bbox'][idx]*100 for r in detail]
  if initial is not None:x=[0]+x;y=[initial[idx]*100]+y
  ax.plot(x,y,'o-',color='#2F5D9B',lw=1.8)
  if not detail:ax.text(.5,.2,'尚无完整训练轮次\n仅展示开训前验证值',ha='center',transform=ax.transAxes,color='#555')
  ax.set_xlim(0,max(2,len(detail)));ax.xaxis.set_major_locator(MaxNLocator(integer=True));style(ax,title,'指标（0–100）')
 ax=axes[0,2]
 ax.barh(['RGB 细节分支','2000 全量微调'],[len(detail),len(full)],color=['#2F5D9B','#697349'])
 for i,n in enumerate([len(detail),len(full)]):ax.text(n+.3,i,f'{n}/20',va='center')
 ax.set_xlim(0,23);ax.set_xlabel('已完成 Epoch');ax.set_title('训练进度',loc='left');ax.grid(axis='x',alpha=.15)
 for ax,key,title in [(axes[1,0],'loss','全量2000：每轮训练损失'),(axes[1,1],'lr','全量2000：检测头学习率')]:
  x=[r['epoch'] for r in full];y=[r['train']['loss'] if key=='loss' else r['lr'][1] for r in full]
  ax.plot(x,y,'o-',color='#697349',markersize=3);style(ax,title,'加权损失' if key=='loss' else '学习率')
  if key=='lr':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
 ax=axes[1,2]
 # Keep interrupted and restarted attempts separate; no invented epoch summaries.
 for epoch in sorted({int(s[0]) for s in step}):
  points=[s for s in step if int(s[0])==epoch]
  # A restart resets batch index. Split into separate attempts.
  segments=[[]]
  for s in points:
   if segments[-1] and int(s[1])<=int(segments[-1][-1][1]):segments.append([])
   segments[-1].append(s)
  for k,seg in enumerate(segments):ax.plot([int(s[1]) for s in seg],[float(s[3]) for s in seg],'o-',markersize=3,label=f'epoch {epoch+1} · 尝试{k+1}')
 if not step:ax.text(.5,.5,'尚无训练 batch 日志',ha='center',transform=ax.transAxes)
 if step:ax.legend(fontsize=8)
 ax.set_title('细节分支：轮内累计平均损失',loc='left');ax.set_xlabel('该轮 Batch index（从0开始）');ax.set_ylabel('加权损失');ax.grid(alpha=.15)
 fig.text(.025,.025,f'{stamp} | 首次尝试因 NCCL 同步超时中断，未完成 epoch。全量2000无独立验证 AP；val400 不能等同 phase2。',fontsize=9,color='#555')
 fig.tight_layout(rect=(0,.07,1,.91))
 tmp=OUT/'overview.tmp.png';fig.savefig(tmp,dpi=150);tmp.replace(OUT/'overview.png');plt.close(fig)
 (OUT/'snapshot.json').write_text(json.dumps({'updated_at':stamp,'detail':detail,'full':full,'queue_status':state,'detail_logged_batches':step},ensure_ascii=False))
 print(json.dumps({'time':stamp,'detail_epochs':len(detail),'full_epochs':len(full),'stage':state['stage']},ensure_ascii=False),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--cached',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  generate(args.cached)
  if not args.watch or all((ROOT/'runs'/n/'COMPLETE').exists() for n in ('detail800','ft2000_aug800')):break
  time.sleep(1200)
