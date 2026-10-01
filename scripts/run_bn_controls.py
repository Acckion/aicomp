"""Persistent, independent single-GPU BN controls with explicit failure state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/mechanism_trials'


def main(job, skip_smoke):
    OUT.mkdir(parents=True, exist_ok=True)
    name, gpu = ('bn_adapt800', 0) if job == 'adapt' else ('bn_frozen800', 1)
    lock = (OUT / f'gpu{gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
           'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
    command = [sys.executable, str(ROOT / 'scripts/train_mechanisms.py'), '--config',
               str(ROOT / f'configs/{name}.yml'), '--init-checkpoint',
               str(ROOT / 'runs/ft_aug800/weights_epoch_020.pth')]

    def status(stage, **details):
        data = {'stage': stage, 'controller_pid': os.getpid(), 'run': name,
                'gpu': gpu, 'time': time.time(), **details}
        path = OUT / f'{name}_status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    try:
        run = ROOT / 'runs' / name
        if (run / 'COMPLETE').exists() or (run / 'metrics.jsonl').exists():
            raise RuntimeError('Run already exists; use explicit resume, never overwrite: ' + name)
        status('waiting_for_memory', required_free_mib=9900)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                       '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                       text=True, timeout=10).strip())
            if free >= 9900:
                break
            time.sleep(10)
        if skip_smoke:
            smoke = OUT / f'{name}_smoke.log'
            if not smoke.exists() or 'SMOKE TEST PASSED' not in smoke.read_text():
                raise RuntimeError('--skip-smoke requires a completed peak-resolution smoke log')
        else:
            status('smoke')
            with (OUT / f'{name}_smoke.log').open('a') as log:
                subprocess.run(command + ['--smoke'], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        with (OUT / f'{name}.log').open('a') as log:
            process = subprocess.Popen(command + ['--evaluate-init'], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status('training', pid=process.pid, command=command + ['--evaluate-init'])
        code = process.wait()
        if code != 0:
            raise RuntimeError(f'Training exited with code {code}; see {name}.log')
        if not (run / 'COMPLETE').exists():
            raise RuntimeError('Training exit without COMPLETE marker')
        status('complete', pid=process.pid)
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=['adapt', 'frozen'], required=True)
    parser.add_argument('--skip-smoke', action='store_true')
    arguments = parser.parse_args()
    main(arguments.job, arguments.skip_smoke)
