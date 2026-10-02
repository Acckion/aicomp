"""Detached GPU8 card2 own-lock queue, with strict preflight before training."""
import fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/ir_corr';PY='/home/fbohan/miniconda3/envs/AICOMP/bin/python'
def main():
 child=None
 signal.signal(signal.SIGTERM,lambda n,f: (_ for _ in ()).throw(SystemExit(128+n)))
 signal.signal(signal.SIGINT,lambda n,f: (_ for _ in ()).throw(SystemExit(128+n)))
 OUT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'experiments/mechanism_trials/gpu2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
 def status(stage,**kw):
  p=OUT/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),'host':'GPU8','gpu':2,'time':time.time(),**kw},indent=2));t.replace(p)
 try:
  def wait_free(next_stage):
   while True:
    free=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free>=9216:break
    status('waiting_memory',next_stage=next_stage,free_mib=free);time.sleep(15)
  wait_free('preflight')
  for stage,args in [('preflight',['--preflight']),('pretraining',['--epochs','4'])]:
   wait_free(stage)
   with (OUT/(stage+'.log')).open('a') as f:child=subprocess.Popen([PY,str(ROOT/'scripts/ir_corr_pretrain.py'),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   status(stage,worker_pid=child.pid);assert child.wait()==0,stage
  gate=json.loads((OUT/'evidence_gate.json').read_text())
  if gate['stage']!='passed':
   status('no_evidence',evidence=gate);return
  def run(stage,command):
   nonlocal child
   wait_free(stage)
   with (OUT/(stage+'.log')).open('a') as f:child=subprocess.Popen([PY,*command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   status(stage,worker_pid=child.pid);assert child.wait()==0,stage
  run('detection_preflight',[str(ROOT/'scripts/ir_corr_detection_preflight.py')])
  assert json.loads((OUT/'detection_preflight.json').read_text())['stage']=='passed'
  for name in ['ir_corr800','ir_corr_control800']:
   command=[str(ROOT/'scripts/train_ir_content.py'),'--config',str(ROOT/'configs'/f'{name}.yml'),'--init-checkpoint',str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')]
   run(name+'_smoke',command+['--smoke']);run(name,command+['--evaluate-init'])
   assert (ROOT/'runs'/name/'COMPLETE').exists()
  status('complete')
 except BaseException:status('failed',error=traceback.format_exc());raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  lock.close()
if __name__=='__main__':main()
