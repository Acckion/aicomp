"""Persistent guarded Co-DINO smoke and training; owns only physical GPU3."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/codino';PYTHON='/dev/shm/aicomp_codino/venv/bin/python';WEIGHT='/dev/shm/aicomp_codino/co_dino_5scale_swin_large_16e_o365tococo.pth'
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'3','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TMPDIR':'/dev/shm/aicomp_codino/tmp','PYTHONUNBUFFERED':'1','CODINO_VENDOR':'/dev/shm/aicomp_codino/vendor/Co-DETR','LD_LIBRARY_PATH':''}

def status(stage,**kw):
 p=OUT/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),'gpu':3,'memory_cap_gib':8.5,'time':time.time(),**kw},ensure_ascii=False,indent=2));t.replace(p)

def free():
 r=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a):int(b) for a,b in (line.split(',') for line in r.splitlines())}[3]

def main():
 OUT.mkdir(exist_ok=True,parents=True);locks=[];child=None
 for name in [OUT/'controller.lock',ROOT/'experiments/mechanism_trials/gpu3.lock']:
  l=name.open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(l)
 def stop(sig,frame):raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  remote=ROOT/'checkpoints/gpu6_storage/codino/runs/codino1600';remote.mkdir(exist_ok=True,parents=True)
  run=ROOT/'runs/codino1600'
  if not run.exists():run.symlink_to(remote,target_is_directory=True)
  assert run.resolve()==remote.resolve()
  while True:
   probe=subprocess.run([PYTHON,'-c','import torch,torchvision,mmcv,mmcv.ops,fairscale,timm; assert torch.__version__.startswith("2.0.0"); assert mmcv.__version__=="1.7.2"'],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
   if probe.returncode==0 and Path(WEIGHT).exists():break
   status('waiting_for_dependencies',error=probe.stderr[-1200:]);time.sleep(10)
  while free()<9500:status('waiting_for_memory',free_mib=free());time.sleep(15)
  command=[PYTHON,str(ROOT/'scripts/train_codino_aic.py'),'--pretrained',WEIGHT,'--epochs','16']
  selected=None
  for size,mode in [(736,'staged'),(640,'staged'),(640,'frozen'),(576,'frozen')]:
   status('smoke',size=size,backbone_mode=mode)
   path=OUT/f'smoke_{size}_{mode}.log'
   with path.open('a') as log:
    child=subprocess.Popen(command+['--smoke','--size',str(size),'--backbone-mode',mode],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   status('smoke',worker_pid=child.pid,size=size,backbone_mode=mode)
   result=child.wait()
   if result==0 and 'SMOKE TEST PASSED' in path.read_text():selected=(size,mode);break
   text=path.read_text()
   if 'out of memory' not in text.lower():raise RuntimeError(f'Smoke failed for a non-memory reason: {path}')
  if not selected:raise RuntimeError('All Swin-L memory profiles failed; R50 pretrained fallback requires preparation')
  size,mode=selected;profile=json.loads((OUT/'smoke_passed.json').read_text());gate=int(profile['peak_allocated_mib'])+1800
  while free()<gate:status('waiting_for_profile_memory',free_mib=free(),required_mib=gate,profile=profile);time.sleep(15)
  atom=OUT/'selected_profile.json';atom.write_text(json.dumps(profile,indent=2))
  while True:
   args=command+['--size',str(size),'--backbone-mode',mode]
   if (run/'last.pth').exists():args+=['--resume',str(run/'last.pth')]
   with (OUT/'train.log').open('a') as log:child=subprocess.Popen(args,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   status('training',worker_pid=child.pid,epochs=16,profile=profile,backbone_mode=mode,size=size,training_images=1600,validation_images=400,metrics=str(run/'metrics.jsonl'))
   rc=child.wait()
   if rc==0:break
   tail=(OUT/'train.log').read_text()[-8000:]
   if 'out of memory' in tail.lower() and mode=='staged' and (run/'last.pth').exists():
    mode='frozen';status('recovering_after_unfreeze_memory',error=tail[-1500:]);time.sleep(10);continue
   raise RuntimeError(f'Training failed rc={rc}: '+tail[-2500:])
  assert (run/'COMPLETE').exists();status('complete',summary=json.loads((run/'summary.json').read_text()))
 except BaseException:
  status('failed',traceback=traceback.format_exc());raise
 finally:
  if child and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
if __name__=='__main__':main()
