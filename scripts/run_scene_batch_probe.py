"""Wait for a project's GPU lock, then compare two memory-bounded batch sizes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-index',type=int,required=True);a=p.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    out=ROOT/'experiments/scene_batch_probe';out.mkdir(parents=True,exist_ok=True)
    own=(out/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/f'experiments/mechanism_trials/gpu{a.gpu_index}.lock').open('a');child=None
    def status(stage,**extra):
        tmp=out/'status.tmp';tmp.write_text(json.dumps({'stage':stage,'time':time.time(),'controller_pid':os.getpid(),**extra},indent=2));tmp.replace(out/'status.json')
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    try:
        while True:
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:status('waiting_existing_task');time.sleep(15)
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=7168:break
            status('waiting_memory',free_mib=free);time.sleep(15)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu_index),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        results={}
        for batch in [1,2]:
            with (out/f'batch{batch}.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/benchmark_scene_batch.py'),'--batch-size',str(batch)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status('probing',batch_size=batch,worker_pid=child.pid)
            code=child.wait()
            if code:
                status('failed_probe',batch_size=batch,returncode=code);return
            results[batch]=json.loads((out/f'batch{batch}.json').read_text())
        comparison={'batch2_speed_ratio':results[2]['images_per_second_last_two_updates']/results[1]['images_per_second_last_two_updates'],
                    'batch1_peak_mib':results[1]['peak_allocated_mib'],'batch2_peak_mib':results[2]['peak_allocated_mib'],
                    'notes':'No automatic migration or evidence of accuracy. Both retain effective batch8.'}
        (out/'comparison.json').write_text(json.dumps(comparison,indent=2));status('complete',comparison=comparison)
    except Exception as e:status('failed',error=repr(e));raise
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()


if __name__=='__main__':main()
