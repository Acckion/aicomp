"""Persistent GPU1 queue: real smoke, instance-memory train, matched control."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/instance_memory';GPU=1
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':str(GPU),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
     'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
def status(stage,**fields):
    p=OUT/'status.json';temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps({'stage':stage,'time':time.time(),'controller_pid':os.getpid(),'gpu':GPU,**fields},indent=2));temp.replace(p)
def free_mib():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b)for a,b in (line.split(',')for line in query.splitlines())}[GPU]
def main():
    OUT.mkdir(parents=True,exist_ok=True);child=None
    job=(OUT/'controller.lock').open('a');fcntl.flock(job,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(ROOT/'experiments/mechanism_trials/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    def run(command,log,stage,**fields):
        nonlocal child
        with (OUT/log).open('a') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage,pid=child.pid,**fields)
        if child.wait():raise RuntimeError(f'{stage} failed: {log}')
    try:
        assert (ROOT/'checkpoints/gpu6_storage/instance_memory/COMPLETE').exists()
        assert json.loads((OUT/'preflight.json').read_text())['stage']=='passed'
        assert subprocess.check_output(['findmnt','-n','-T',str(ROOT/'checkpoints/gpu6_storage'),'-o','FSTYPE'],text=True).strip()=='fuse.sshfs'
        for name in ['instance_memory800','instance_memory_control800']:
            remote=ROOT/'checkpoints/gpu6_storage/instance_memory/runs'/name;remote.mkdir(parents=True,exist_ok=True)
            path=ROOT/'runs'/name
            if path.is_symlink():assert path.resolve()==remote.resolve()
            elif path.exists():raise RuntimeError(f'Existing non-symlink {path}')
            else:path.symlink_to(remote,target_is_directory=True)
        while free_mib()<7800:status('waiting_for_memory',free_mib=free_mib());time.sleep(15)
        parent=ROOT/'runs/ft_aug800/weights_epoch_020.pth'
        for name in ['instance_memory800','instance_memory_control800']:
            output=ROOT/'runs'/name
            if (output/'COMPLETE').exists():continue
            if (output/'metrics.jsonl').exists():raise RuntimeError(f'Partial inference-only run cannot resume {name}')
            command=[sys.executable,str(ROOT/'scripts/train_instance_memory.py'),'--config',str(ROOT/'configs'/f'{name}.yml'),
                     '--init-checkpoint',str(parent)]
            run(command+['--smoke'],f'{name}_smoke.log','smoke',run=name)
            assert 'SMOKE TEST PASSED' in (OUT/f'{name}_smoke.log').read_text()
            run(command+['--evaluate-init'],f'{name}.log','training',run=name,epochs=8,
                effective_batch=12,per_gpu_batch=2,memory_limit_gib=8.5,
                queued_after_this=['instance_memory_control800','paired_400_evaluation'] if name=='instance_memory800' else ['paired_400_evaluation'])
            assert (output/'COMPLETE').exists()
        run([sys.executable,str(ROOT/'scripts/evaluate_instance_memory.py')],'paired_evaluation.log','paired_evaluation')
        status('complete',runs=['instance_memory800','instance_memory_control800'],report=str(OUT/'paired_report.json'))
    except BaseException:status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        lock.close();job.close()
if __name__=='__main__':main()
