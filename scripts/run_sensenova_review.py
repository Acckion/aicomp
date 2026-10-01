"""Start the local region-review pilot without interrupting existing jobs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/sensenova_review'
RAM=Path('/dev/shm/aicomp_sensenova')
PYTHON='/home/fbohan/miniconda3/envs/AICOMP/bin/python'
GPUS=[1,5,6,7]

def status(stage,**extra):
    temp=OUT/'controller_status.json.tmp'
    temp.write_text(json.dumps({'stage':stage,'time':time.time(),'gpus':GPUS,**extra},indent=2))
    temp.replace(OUT/'controller_status.json')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (RAM/'download.complete').exists()
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
        if all(free[g]>=9900 for g in GPUS):break
        status('waiting_for_gpu_headroom',free_mib=free);time.sleep(30)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS)),OMP_NUM_THREADS='2',
        PYTHONPATH=str(RAM/'deps')+':'+str(RAM/'source'),HF_HOME=str(RAM/'hf'),TMPDIR=str(RAM),
        LD_LIBRARY_PATH='/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib',PYTHONUNBUFFERED='1')
    with (OUT/'worker.log').open('ab') as log:
        worker=subprocess.Popen([PYTHON,'-u',str(ROOT/'scripts/sensenova_review.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        (OUT/'worker.pid').write_text(str(worker.pid));status('worker_running',pid=worker.pid)
        code=worker.wait()
    if code:raise RuntimeError(f'Review worker exited {code}; see worker.log')
    status('complete')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=repr(exc));traceback.print_exc();raise
