"""Own one GPU, validate real peak memory, run paired/control and IR ablations."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/ir_query_alignment'
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
     'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}

def main():
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'experiments/mechanism_trials/gpu2.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    def status(stage,**detail):
        path=OUT/'status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),'time':time.time(),'gpu':2,**detail},indent=2));tmp.replace(path)
    def run(command,log,stage,**detail):
        nonlocal child
        with (OUT/log).open('ab')as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            status(stage,pid=child.pid,command=command,**detail)
            code=child.wait()
            if code:raise RuntimeError(f'{stage} failed: exit {code}, log {log}')
    def terminate(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        filesystem=subprocess.check_output(['findmnt','-n','-T',str(ROOT/'checkpoints/gpu6_storage'),'-o','FSTYPE'],text=True).strip()
        assert filesystem=='fuse.sshfs'
        for job in ['ir_query800','ir_query_control800']:
            target=ROOT/'checkpoints/gpu6_storage/ir_query_alignment/runs'/job
            target.mkdir(parents=True,exist_ok=True)
            output=ROOT/'runs'/job
            if output.is_symlink():assert output.resolve()==target.resolve()
            elif output.exists():raise RuntimeError('Existing non-symlink experiment output')
            else:output.symlink_to(target,target_is_directory=True)
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=9500:break
            status('waiting_for_memory',free_mib=free,required_free_mib=9500);time.sleep(15)
        run([sys.executable,str(ROOT/'scripts/preflight_ir_query.py'),'--gpu'],'preflight.log','preflight')
        preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['stage']=='passed'
        for job in ['ir_query800','ir_query_control800']:
            config=ROOT/f'configs/{job}.yml';output=ROOT/'runs'/job
            if (output/'COMPLETE').exists():continue
            command=[sys.executable,str(ROOT/'scripts/train_mechanisms.py'),'--config',str(config)]
            if (output/'last.pth').exists():
                command.extend(['--resume',str(output/'last.pth')])
            elif (output/'best.pth').exists() and (output/'initial_metrics.json').exists():
                # Initial full state is resumable, preserving epoch0 selection
                # after a failure before the first optimizer step.
                command.extend(['--resume',str(output/'best.pth')])
            else:
                command.extend(['--init-checkpoint',str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')])
                run(command+['--smoke'],f'{job}_smoke.log','smoke',job=job)
            run(command+['--evaluate-init'],f'{job}.log','training',job=job,epochs=8)
            assert (output/'COMPLETE').exists()
            if job=='ir_query800':
                run([sys.executable,str(ROOT/'scripts/evaluate_ir_query_ablations.py'),'--checkpoint',str(output/'best.pth')],
                    'ablations.log','ablations',job=job)
        status('complete')
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll()is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            except ProcessLookupError:pass

if __name__=='__main__':main()
