"""Locked overnight experimental run followed by an identical-budget control."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/aux_detection'
GPU = 7


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'experiments/mechanism_trials/gpu7.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU), 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
           'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
    def status(stage, **details):
        path = OUT/'status.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'stage':stage, 'time':time.time(), 'gpu':GPU,
                                   'controller_pid':os.getpid(), **details}, indent=2))
        temp.replace(path)
    def wait_memory(minimum=9300):
        while True:
            free = int(subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'],text=True).strip())
            if free >= minimum: return
            status('waiting_for_memory', free_mib=free, minimum_free_mib=minimum)
            time.sleep(20)
    def execute(command, log_name, stage, **details):
        with (OUT/log_name).open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
            status(stage, worker_pid=process.pid, log=str(OUT/log_name), **details)
            code = process.wait()
            if code: raise RuntimeError(f'{stage} failed with exit code {code}')
    try:
        wait_memory()
        execute([sys.executable, str(ROOT/'scripts/check_aux_detection.py')], 'numerical_check.log', 'checking')
        for name in ['aux_detection800', 'aux_detection_control800']:
            run = ROOT/'runs'/name
            if (run/'COMPLETE').exists(): continue
            if (run/'metrics.jsonl').exists(): raise RuntimeError(f'Refuse to overwrite partial {name}')
            wait_memory()
            command = [sys.executable,str(ROOT/'scripts/train_aux_detection.py'),'--config',
                       str(ROOT/f'configs/{name}.yml'),'--init-checkpoint',
                       str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')]
            execute(command+['--smoke'],name+'_smoke.log','smoke',run=name)
            execute(command,name+'.log','training',run=name,output=str(run),control_queued=name=='aux_detection800')
            if not (run/'COMPLETE').exists(): raise RuntimeError(f'No COMPLETE for {name}')
            status('run_complete', run=name)
        for name in ['aux_detection800', 'aux_detection_control800']:
            execute([sys.executable,str(ROOT/'scripts/evaluate_aux_detection.py'),'--run',name],
                    name+'_evaluation.log','evaluating',run=name)
        status('complete')
    except BaseException:
        status('failed',traceback=traceback.format_exc())
        raise


if __name__ == '__main__': main()
