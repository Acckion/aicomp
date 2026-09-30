"""Run the previously skipped, strictly matched crop continuation control."""
import os,sys,time,json,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/crop_control'
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def status(stage,**details):
 p=OUT/'status.tmp';p.write_text(json.dumps({'time':time.strftime('%F %T'),'stage':stage,**details},indent=2));p.replace(OUT/'status.json')


def available():
 q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
 m={int(x[0]):int(x[1]) for line in q.splitlines() if (x:=line.split(','))}
 return all(m[g]>=9900 for g in [0,2,5])


def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status('waiting_for_standard_lr_trials',gpus=[0,2,5],dependency='tune800_high_standard',epochs=15)
 while not (ROOT/'runs/tune800_high_standard/COMPLETE').exists():
  p=ROOT/'experiments/light_tuning/status.json'
  if p.exists() and json.loads(p.read_text()).get('stage')=='failed':raise RuntimeError('LR trial queue failed; inspect before acquiring its GPUs')
  time.sleep(5)
 while not available():time.sleep(5)
 # This controller skips the already completed crop run and makes both reports.
 with (OUT/'train.log').open('a') as log:
  p=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_targetcrop_stage.py')],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status('training',controller_pid=p.pid,gpus=[0,2,5])
 if p.wait():raise RuntimeError('Matched continuation control failed')
 sys.path.insert(0,str(ROOT/'scripts'))
 from after_targetcrop import choose,records
 ann=json.loads((ROOT/'data/annotations/val400.json').read_text())
 counts={c['name']:sum(a['category_id']==c['id'] for a in ann['annotations']) for c in ann['categories']}
 initial=json.loads((ROOT/'runs/targetcrop800/initial_metrics.json').read_text())['coco_eval_bbox']
 decision=choose(records('targetcrop800'),records('continue800_control'),initial,counts)
 (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
 status('comparison_ready',decision=decision,report=str(ROOT/'experiments/targetcrop/comparison.json'),note='No structural experiment restart; results will guide crop full-data transfer separately.')


if __name__=='__main__':
 try:main()
 except Exception:
  status('failed',traceback=traceback.format_exc());raise
