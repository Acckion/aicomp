"""Minute-refresh pretraining and independent detector monitoring."""
import json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/ir_corr';OUT.mkdir(parents=True,exist_ok=True)
def plot():
 progress=ROOT/'experiments/ir_corr/pretraining.log';rows=[]
 if progress.exists():
  for line in progress.read_text(errors='replace').splitlines():
   try:r=json.loads(line)
   except Exception:continue
   if isinstance(r,dict) and 'steps' in r:rows.append(r)
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for ax,keys,title in [(axes[0,0],['loss','intra_loss','cross_loss'],'Train-only objective'),(axes[0,1],['normalized_endpoint_error','recall_1cell'],'Known intra-transform retrieval'),(axes[1,0],['cycle','wrong_image_cycle'],'Cycle consistency (not alignment GT)')]:
  for key in keys:
   vals=[r for r in rows if key in r]
   if vals:ax.plot([r['steps'] for r in vals],[r[key] for r in vals],label=key)
  ax.set_title(title);ax.set_xlabel('optimizer step');ax.grid(alpha=.3)
  if ax.lines:ax.legend()
 ax=axes[1,1]
 for name in ['ir_corr800','ir_corr_control800']:
  path=ROOT/'runs'/name/'metrics.jsonl'
  if path.exists():
   vals=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
   aps=[(r.get('epoch',j+1),r.get('validation',{}).get('coco_eval_bbox',[0])[0]*100) for j,r in enumerate(vals)]
   ax.plot(*zip(*aps),label=name)
 ax.set_title('Independent validation mAP@50-95 (when detector starts)');ax.grid(alpha=.3)
 if ax.lines:ax.legend()
 fig.tight_layout();fig.savefig(OUT/'overview.png');plt.close(fig)
while True:
 try:plot()
 except Exception as e:print(type(e).__name__,str(e),flush=True)
 time.sleep(60)
