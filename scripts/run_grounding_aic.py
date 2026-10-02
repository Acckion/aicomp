"""GPU0 conservative admission and independent persistent GroundingDINO launcher."""
import os,json,sys,time,subprocess,fcntl,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/grounding';OUT.mkdir(exist_ok=True)
lock=open(OUT/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
child=None
def stop(signum,frame):
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 sys.exit(128+signum)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
(OUT/'controller.pid').write_text(str(os.getpid()))
while True:
 free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free>=9216:break
 (OUT/'status.json').write_text(json.dumps({'state':'waiting_for_memory','gpu':0,'free_mib':free,'required_mib':9216,'time':time.time()}));time.sleep(30)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR='/dev/shm/aicomp_grounding/tmp',PYTHONUNBUFFERED='1')
py='/dev/shm/aicomp_grounding/venv/bin/python'
resume=(ROOT/'checkpoints/gpu6_storage/grounding/grounding1600/last_checkpoint').exists()
for stage,args in [('smoke',['--smoke']),('train',['--resume'] if resume else [])]:
 (OUT/'status.json').write_text(json.dumps({'state':stage,'stage':stage,'run':'grounding1600','gpu':0,'time':time.time()}))
 with open(OUT/(stage+'.log'),'a') as log:
  child=subprocess.Popen([py,str(ROOT/'scripts/train_grounding_aic.py'),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(OUT/(stage+'.pid')).write_text(str(child.pid));code=child.wait()
 if code:
  (OUT/'status.json').write_text(json.dumps({'state':'failed','stage':stage,'returncode':code,'time':time.time()}));sys.exit(code)
