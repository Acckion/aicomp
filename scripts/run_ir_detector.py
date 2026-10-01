"""Persistent two-card IR adaptation, with real DDP smoke before training."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/multimodal_probe'
GPUS=[5,6]
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,GPUS)),
     'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false',
     'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}

def write(stage,**fields):
    p=OUT/'ir_detector_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'stage':stage,'gpus':GPUS,'time':time.time(),
                              'controller_pid':os.getpid(),**fields},ensure_ascii=False,indent=2))
    tmp.replace(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    locks=[]
    for path in [OUT/'ir_detector.lock',*(ROOT/f'experiments/mechanism_trials/gpu{g}.lock' for g in GPUS)]:
        lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    child=None
    def stop(number,frame):raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['stage']=='passed'
        mount=ROOT/'checkpoints/gpu6_storage'
        assert subprocess.check_output(['findmnt','-n','-T',str(mount),'-o','FSTYPE'],text=True).strip()=='fuse.sshfs'
        remote=mount/'multimodal_probe/runs/ir_detector800';remote.mkdir(parents=True,exist_ok=True)
        output=ROOT/'runs/ir_detector800'
        if output.is_symlink():assert output.resolve()==remote.resolve()
        elif output.exists():raise RuntimeError('Refusing to replace an existing run')
        else:output.symlink_to(remote,target_is_directory=True)
        if (output/'metrics.jsonl').exists() or (output/'COMPLETE').exists():
            raise RuntimeError('Refusing to overwrite a started/completed experiment')
        while True:
            query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(a):int(b) for a,b in (line.split(',') for line in query.splitlines())}
            if all(free[g]>=9900 for g in GPUS):break
            write('waiting_for_memory',free_mib={g:free[g] for g in GPUS});time.sleep(15)
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                 str(ROOT/'scripts/train_ir_detector.py'),'--config',str(ROOT/'configs/ir_detector800.yml'),
                 '--init-checkpoint',str(ROOT/'checkpoints/gpu6_storage/multimodal_probe/ir_initialization.pth')]
        with (OUT/'ir_detector_smoke.log').open('a') as log:
            child=subprocess.Popen(command+['--smoke'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write('smoke',pid=child.pid)
            if child.wait():raise RuntimeError('IR peak-resolution DDP smoke failed')
        assert 'SMOKE TEST PASSED' in (OUT/'ir_detector_smoke.log').read_text()
        with (OUT/'ir_detector.log').open('a') as log:
            child=subprocess.Popen(command+['--evaluate-init'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('training',pid=child.pid,epochs=24,training_images=1600,validation_images=400,
              per_gpu_memory_limit_gib=8.5,per_gpu_batch=2,effective_batch=12,
              purpose='Learn detection on IR and measure unique coverage vs frozen RGB; no deployed detection ensemble.')
        if child.wait():raise RuntimeError('IR training failed')
        assert (output/'COMPLETE').exists()
        write('complete',pid=child.pid)
    except BaseException:
        write('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            except ProcessLookupError:pass

if __name__=='__main__':main()
