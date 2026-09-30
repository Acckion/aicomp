"""Two LR x two augmentation trials, plus same-model horizontal TTA/Soft-NMS."""
import os,sys,time,json,subprocess,fcntl,traceback,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/light_tuning'
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
START=ROOT/'runs/ft_aug800/weights_epoch_020.pth'


def status(stage,**details):
 p=OUT/'status.tmp';p.write_text(json.dumps({'time':time.strftime('%F %T'),'stage':stage,**details},indent=2));p.replace(OUT/'status.json')


def free(gpus):
 q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
 m={int(v[0]):int(v[1]) for line in q.splitlines() if (v:=line.split(','))}
 return all(m[g]>=9900 for g in gpus)


def spawn(cmd,gpus,label):
 with (OUT/(label+'.log')).open('a') as f:
  return subprocess.Popen(cmd,cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)


def report(name):
 rows=[json.loads(s) for s in (ROOT/'runs'/name/'metrics.jsonl').read_text().splitlines()]
 best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0]);initial=json.loads((ROOT/'runs'/name/'initial_metrics.json').read_text())['coco_eval_bbox']
 return {'name':name,'best_epoch':best['epoch'],'best_map':best['validation']['coco_eval_bbox'][0]*100,'best_stats':best['validation']['coco_eval_bbox'],'best_classes':best['validation']['per_class_ap'],'initial_stats':initial,'initial_map':initial[0]*100,'gain_from_initial':(best['validation']['coco_eval_bbox'][0]-initial[0])*100,'median_last3_map':statistics.median(r['validation']['coco_eval_bbox'][0]*100 for r in rows[-3:]),'last_epoch':rows[-1]['epoch']}


def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 groups=[([0,2,5],['tune800_low_standard','tune800_high_standard']),([1,3,6],['tune800_low_mild','tune800_high_mild'])]
 # Cache reuse preserves the original result files and avoids repeated plain inference.
 plain=OUT/'infer_plain';plain.mkdir(exist_ok=True)
 for filename in ('raw_predictions.json','cache_identity.json','unclipped_reference.json'):
  src=ROOT/'experiments/next_stage/infer800_tile0'/filename
  if src.exists() and not (plain/filename).exists():shutil.copy2(src,plain/filename)
 inference_pending=[('infer_plain',False),('infer_flip',True)];inference_active=None
 active={};reports=[];eval_pending=[];eval_active=None;inference_results=[]
 status('starting',groups=[g for g,_ in groups])
 while any(q for g,q in groups) or active or inference_pending or inference_active or eval_pending or eval_active:
  for key,(p,name,phase,cmd,gpus) in list(active.items()):
   code=p.poll()
   if code is None:continue
   if code:raise RuntimeError(f'{name} {phase} failed ({code})')
   if phase=='smoke':
    p=spawn(cmd+['--evaluate-init'],gpus,name);active[key]=(p,name,'training',cmd,gpus)
   else:
    if not (ROOT/'runs'/name/'COMPLETE').exists():raise RuntimeError(f'{name} missing COMPLETE')
    r=report(name);reports.append(r);(OUT/'training_results.json').write_text(json.dumps(reports,indent=2));eval_pending.append((name,ROOT/f"runs/{name}/weights_epoch_{r['best_epoch']:03d}.pth"));del active[key]
  if inference_active and inference_active[0].poll() is not None:
   p,label=inference_active
   if p.returncode:raise RuntimeError(f'Inference {label} failed')
   inference_results.append(json.loads((OUT/label/'results.json').read_text()));inference_active=None
  if eval_active and eval_active[0].poll() is not None:
   p,label=eval_active
   if p.returncode:raise RuntimeError(f'Candidate evaluation {label} failed')
   inference_results.append(json.loads((OUT/label/'results.json').read_text()));eval_active=None
  if not inference_active and not eval_active and free([7]):
   if inference_pending:
    label,flip=inference_pending.pop(0)
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(START),'--size','800','--expanded-soft','--output',str(OUT/label)]
    if flip:cmd.append('--flip-tta')
    inference_active=(spawn(cmd,[7],label),label)
   elif eval_pending:
    name,weights=eval_pending.pop(0);label='infer_'+name
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(weights),'--size','800','--expanded-soft','--flip-tta','--output',str(OUT/label)]
    eval_active=(spawn(cmd,[7],label),label)
  for key,(gpus,queue) in enumerate(groups):
   if key in active or not queue or not free(gpus):continue
   name=queue.pop(0)
   if (ROOT/'runs'/name/'metrics.jsonl').exists():raise RuntimeError(f'{name} exists; explicit resume required')
   cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3',f'--master_port={29941+key}',str(ROOT/'scripts/train_baseline.py'),'--config',str(ROOT/f'configs/{name}.yml'),'--init-checkpoint',str(START)]
   active[key]=(spawn(cmd+['--smoke'],gpus,name+'_smoke'),name,'smoke',cmd,gpus)
  status('running',jobs={str(k):{'name':a[1],'phase':a[2],'pid':a[0].pid,'gpus':a[4]} for k,a in active.items()},finished_training=[r['name'] for r in reports],inference_pid=inference_active[0].pid if inference_active else (eval_active[0].pid if eval_active else None))
  time.sleep(5)
 combined=[]
 for d in inference_results:
  for row in d['results']:combined.append({'checkpoint':d['checkpoint'],'flip_tta':d.get('flip_tta',False),'size':d['size'],**row})
 base=next(r for r in combined if r['checkpoint']==str(START) and not r['flip_tta'] and r['method']=='none')
 for r in combined:r['gain_from_same_start_plain']=r['map']-base['map']
 combined.sort(key=lambda r:r['map'],reverse=True)
 (OUT/'comparison.json').write_text(json.dumps({'training':reports,'inference':combined,'reference':base,'note':'val400 comparison, no phase2 score; no model/weight ensemble; four training trials share a starting checkpoint, seed, batch and duration.'},indent=2))
 status('comparison_ready',report=str(OUT/'comparison.json'),leaderboard_score=None)


if __name__=='__main__':
 try:main()
 except Exception:
  status('failed',traceback=traceback.format_exc());raise
