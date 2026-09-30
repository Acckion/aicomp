"""Refresh LR/augmentation trial curves and available TTA/Soft-NMS comparisons."""
import argparse,fcntl,json,time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import plot_targetcrop as charts
ROOT=Path(__file__).resolve().parents[1]
charts.OUT=ROOT/'monitoring/light_tuning'
charts.RUNS={
 'tune800_low_standard':('LR5e-6 · 原增强','#2F5D9B','-',8),
 'tune800_low_mild':('LR5e-6 · 温和增强','#2F5D9B','--',8),
 'tune800_high_standard':('LR1.5e-5 · 原增强','#C47B32','-',8),
 'tune800_high_mild':('LR1.5e-5 · 温和增强','#C47B32','--',8),
}


def generate():
 charts.generate()
 rows=[]
 for name in ['infer_plain','infer_flip']:
  p=ROOT/'experiments/light_tuning'/name/'results.json'
  if not p.exists():continue
  try:d=json.loads(p.read_text())
  except json.JSONDecodeError:continue
  for r in d['results']:rows.append((name,r))
 if not rows:return
 labels=[('原图' if n=='infer_plain' else '原图+翻转')+' · '+r['method']+f" {r['threshold']:g}" for n,r in rows]
 fig,axes=plt.subplots(1,2,figsize=(16,max(7,len(rows)*.32)))
 colors=['#2F5D9B' if n=='infer_plain' else '#C47B32' for n,r in rows]
 for ax,idx,title in [(axes[0],0,'mAP@50–95'),(axes[1],3,'小目标 AP')]:
  values=[r['stats'][idx] for n,r in rows]
  ax.barh(range(len(rows)),values,color=colors)
  ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlim(0,100)
  ax.set_title(title,loc='left');ax.set_xlabel('val400 AP（0–100）');ax.grid(axis='x',alpha=.15)
  for i,v in enumerate(values):ax.text(v+.4,i,f'{v:.2f}',va='center',fontsize=9)
 stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
 fig.suptitle('同一800权重 · 水平翻转 TTA / Soft-NMS 对照',fontsize=17)
 fig.text(.02,.012,f'{stamp} | soft 为线性IoU阈值；soft_gaussian 参数为 sigma。最终最多100框；本地验证不代表 phase2。',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,.96));tmp=charts.OUT/'inference.tmp.png';fig.savefig(tmp,dpi=130);tmp.replace(charts.OUT/'inference.png');plt.close(fig)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
 charts.OUT.mkdir(parents=True,exist_ok=True)
 lock=(charts.OUT/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  generate()
  p=ROOT/'experiments/light_tuning/status.json'
  state=json.loads(p.read_text()) if p.exists() else {}
  if not args.watch or state.get('stage') in ('comparison_ready','failed'):break
  time.sleep(60)
