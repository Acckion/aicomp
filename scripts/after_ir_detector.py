"""Automatically measure useful IR coverage at fixed detector checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/multimodal_probe'
GPU=int(os.environ.get('AICOMP_IR_EVAL_GPU','3'))
MIN_FREE_MIB=int(os.environ.get('AICOMP_IR_EVAL_MIN_FREE_MIB','9900'))
EVAL_MEMORY_GIB=float(os.environ.get('AICOMP_IR_EVAL_MEMORY_GIB','8.5'))
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':str(GPU),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
     'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
def write(stage,**fields):
    p=OUT/'complement_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'stage':stage,'gpu':GPU,'time':time.time(),**fields},ensure_ascii=False,indent=2));tmp.replace(p)
def main():
    lock=(OUT/'complement.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for epoch in [3,6,12,24]:
            checkpoint=ROOT/f'runs/ir_detector800/weights_epoch_{epoch:03d}.pth'
            while not checkpoint.exists():
                status_path=OUT/'ir_detector_status.json'
                if status_path.exists() and json.loads(status_path.read_text()).get('stage')=='failed':
                    raise RuntimeError('IR training failed before selected checkpoint')
                write('waiting_for_checkpoint',epoch=epoch);time.sleep(30)
            output=OUT/f'ir_epoch_{epoch:03d}'
            if (output/'COMPLETE').exists():continue
            gpu_lock=(ROOT/f'experiments/mechanism_trials/gpu{GPU}.lock').open('a')
            while True:
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:write('waiting_for_gpu_lock',epoch=epoch);time.sleep(30)
            try:
                while True:
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free>=MIN_FREE_MIB:break
                    write('waiting_for_memory',epoch=epoch,free_mib=free);time.sleep(30)
                with (OUT/f'ir_epoch_{epoch:03d}.log').open('a') as log:
                    process=subprocess.Popen([sys.executable,str(ROOT/'scripts/evaluate_ir_complementarity.py'),
                      '--checkpoint',str(checkpoint),'--output',str(output),
                      '--memory-gib',str(EVAL_MEMORY_GIB)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    write('evaluating',epoch=epoch,pid=process.pid)
                    if process.wait():raise RuntimeError(f'IR complement evaluation failed at epoch{epoch}')
                assert (output/'COMPLETE').exists()
                write('epoch_complete',epoch=epoch,report=str(output/'report.json'))
            finally:gpu_lock.close()
        write('complete')
    except BaseException:
        write('failed',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
