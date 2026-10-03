"""Minute-updated paired continuation plot; epoch0 is the heldout initialization."""
import fcntl,json,os,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/native_encoded'
NAMES=['scene_native_encoded_delta800','scene_native_encoded_control800','scene_native_routed_delta800']
LABELS={NAMES[0]:'Fixed .6 regions',NAMES[1]:'Continuation control',NAMES[2]:'Routed .3 regions'}

def refresh():
 runs={};evidence={'updated_at':time.strftime('%F %T'),'parent_epoch':20,'validation':'Independent grouped390; not phase2','runs':{},'paired':None,'paired_by_run':{},'limitations':'Only same continuation epochs are comparable. Allocation caps do not change the recipe. Initialization is epoch0; epoch1 is warmup. Small class counts can move macro AP.'}
 for n in NAMES:
  path=ROOT/'runs'/n;rows=[]
  if (path/'initial_metrics.json').exists():rows.append({'epoch':0,'validation':json.loads((path/'initial_metrics.json').read_text())})
  if (path/'metrics.jsonl').exists():
   for line in (path/'metrics.jsonl').read_text().splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
  runs[n]=rows;complete=[r for r in rows if r['epoch']>0]
  evidence['runs'][n]={'completed_epochs':complete[-1]['epoch'] if complete else 0,'map':100*rows[-1]['validation']['coco_eval_bbox'][0] if rows else None}
 b={r['epoch']:r for r in runs[NAMES[1]] if r['epoch']>=2}
 for name in [NAMES[0],NAMES[2]]:
  a={r['epoch']:r for r in runs[name] if r['epoch']>=2};common=sorted(a.keys()&b.keys())[-3:]
  if len(common)==3:
   ds=[100*(a[e]['validation']['coco_eval_bbox'][0]-b[e]['validation']['coco_eval_bbox'][0]) for e in common]
   comparison={'epochs':common,'map_deltas':ds,'mean_map_delta':sum(ds)/3,'status':'Descriptive matched comparison, no automatic promotion'}
   evidence['paired_by_run'][name]=comparison
   if name==NAMES[0]:evidence['paired']=comparison
 fig,axes=plt.subplots(2,3,figsize=(14,8))
 for n,rows in runs.items():
  label=LABELS[n]
  for ax,index,title in zip(list(axes.flat)[:5],[0,1,2,3,None],['mAP@50-95','AP50','AP75','Small AP','AP90']):
   if index is None:
    pairs=[(r['epoch'],100*r['validation']['ap_by_iou']['0.90']) for r in rows if r['validation'].get('ap_by_iou',{}).get('0.90') is not None]
   else:
    pairs=[(r['epoch'],100*r['validation']['coco_eval_bbox'][index]) for r in rows if len(r['validation'].get('coco_eval_bbox',[]))>index]
   if pairs:ax.plot(*zip(*pairs),marker='o',label=label)
   ax.set_title(title)
  pairs=[(r['epoch'],r['train']['loss']) for r in rows if 'loss' in r.get('train',{})]
  if pairs:axes.flat[5].plot(*zip(*pairs),marker='o',label=label)
 axes.flat[5].set_title('Training loss')
 for ax in axes.flat:
  ax.set_xlabel('Continuation epoch');ax.grid(alpha=.25);ax.axvspan(0,1,color='gray',alpha=.08)
  if ax.lines:ax.legend(fontsize=9)
 fig.suptitle('Same parent@20, packed RGB recipe and effective batch8 | not phase2 | '+evidence['updated_at'])
 fig.tight_layout();p=OUT/f'overview.{os.getpid()}.png';fig.savefig(p,dpi=130);plt.close(fig);p.replace(OUT/'overview.png')
 p=OUT/'evidence.tmp';p.write_text(json.dumps(evidence,indent=2));p.replace(OUT/'evidence.json')

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);owner=(OUT/'monitor.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:refresh()
  except (OSError,ValueError) as error:print('Observation retry',repr(error),flush=True)
  time.sleep(60)
