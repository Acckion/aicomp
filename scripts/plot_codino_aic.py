"""Refresh Co-DINO metrics/status every 60 seconds, without invented initial mAP."""
import json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/codino';OUT.mkdir(exist_ok=True,parents=True)
def plot():
 path=ROOT/'runs/codino1600/metrics.jsonl';rows=[]
 if path.exists():
  for line in path.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 fig,axes=plt.subplots(2,2,figsize=(13,9));epochs=[r['epoch'] for r in rows]
 for ax,key,title in zip(axes.flat,['map','ap75','aps','ap90'],['mAP @50:95','AP75','AP small','AP90']):
  ax.plot(epochs,[r['validation'][key] for r in rows],marker='o');ax.set_title(title);ax.set_xlabel('epoch');ax.set_ylabel('AP (%)');ax.grid(alpha=.3)
 status_path=ROOT/'experiments/codino/status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {'stage':'not started'}
 progress_path=ROOT/'runs/codino1600/optimizer_progress.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
 fig.suptitle('Single Co-DINO Swin-L / train1600 + val400\n'+status['stage']+' / '+str(progress.get('epoch','-'))+' epoch / '+str(progress.get('batch','-'))+' batch; '+str(len(rows))+' completed validation epochs')
 fig.tight_layout();tmp=OUT/'overview.tmp.png';fig.savefig(tmp,dpi=140);tmp.replace(OUT/'overview.png');plt.close(fig)
 (OUT/'status.json').write_text(json.dumps({'training':status,'progress':progress,'completed_epochs':len(rows),'updated_at':time.time()},indent=2))
while True:
 try:plot()
 except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
 time.sleep(60)
