"""Train real P2 features alongside the matched crop continuation control."""
import os,sys,time,json,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/p2'
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'1,3,6','OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def status(stage,**details):
 p=OUT/'status.tmp';p.write_text(json.dumps({'time':time.strftime('%F %T'),'stage':stage,**details},indent=2));p.replace(OUT/'status.json')


def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'runs/p2_640/metrics.jsonl').exists():raise RuntimeError('P2 run already exists; explicit resume required')
 status('waiting_for_memory',gpus=[1,3,6])
 while True:
  q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
  m={int(x[0]):int(x[1]) for line in q.splitlines() if (x:=line.split(','))}
  if all(m[g]>=9900 for g in [1,3,6]):break
  time.sleep(5)
 cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3','--master_port=29961',str(ROOT/'scripts/train_baseline.py'),'--config',str(ROOT/'configs/p2_640.yml'),'--init-checkpoint',str(ROOT/'checkpoints/p2_from_ft800.pth')]
 status('smoke',gpus=[1,3,6])
 with (OUT/'smoke.log').open('a') as log:subprocess.run(cmd+['--smoke'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (OUT/'train.log').open('a') as log:p=subprocess.Popen(cmd+['--evaluate-init'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status('training',pid=p.pid,epochs=20,gpus=[1,3,6])
 if p.wait():raise RuntimeError('P2 training failed')
 if not (ROOT/'runs/p2_640/COMPLETE').exists():raise RuntimeError('P2 missing COMPLETE')
 status('training_complete',note='Independent val400 screening needed before full-data transfer; phase2 score unknown')


if __name__=='__main__':
 try:main()
 except Exception:
  status('failed',traceback=traceback.format_exc());raise
