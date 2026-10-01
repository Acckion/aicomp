"""Four persistent jobs: full-data transfer plus isolated matching/ranking trials."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/next_mechanisms'
JOBS = {
    'full2000_bn800': {'gpu': 0, 'parent': 'ft2000_aug800', 'validation': False},
    'full2000_dense800': {'gpu': 3, 'parent': 'ft2000_aug800', 'validation': False},
    'highorder800': {'gpu': 5, 'parent': 'ft_aug800', 'validation': True},
    'queryrank800': {'gpu': 6, 'parent': 'ft_aug800', 'validation': True},
}
ENV = {**os.environ, 'OMP_NUM_THREADS': '2',
       'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main(name):
    OUT.mkdir(parents=True, exist_ok=True)
    job = JOBS[name]
    gpu = job['gpu']
    lock = (OUT / f'{name}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = (ROOT / 'experiments/mechanism_trials' / f'gpu{gpu}.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    spawning = False
    pending_signal = None

    def stop_signal(number, frame):
        nonlocal pending_signal
        if spawning:
            pending_signal = number
        else:
            raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def status(stage, **details):
        write_json(OUT / f'{name}_status.json',
                   {'stage': stage, 'run': name, 'gpu': gpu, 'controller_pid': os.getpid(),
                    'time': time.time(), 'validation': job['validation'], **details})

    def spawn(command, log):
        nonlocal child, spawning
        spawning = True
        try:
            child = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            spawning = False
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)
        return child

    try:
        from run_dense_supervision import active_workers
        config = ROOT / f'configs/{name}.yml'
        output = ROOT / 'runs' / name
        if active_workers(config):
            raise RuntimeError('A detached worker is already running this config')
        if (output / 'COMPLETE').exists() or (output / 'metrics.jsonl').exists():
            raise RuntimeError('Refusing to overwrite an existing training run')
        parent = ROOT / 'runs' / job['parent'] / 'weights_epoch_020.pth'
        if not parent.exists():
            raise FileNotFoundError(parent)
        command = [sys.executable, str(ROOT / 'scripts/train_mechanisms.py'),
                   '--config', str(config), '--init-checkpoint', str(parent)]
        status('waiting_for_memory', minimum_free_mib=9900)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                       '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                       text=True, timeout=10).strip())
            if free >= 9900:
                break
            time.sleep(10)
        status('smoke')
        with (OUT / f'{name}_smoke.log').open('a') as log:
            if spawn(command + ['--smoke'], log).wait():
                raise RuntimeError('Peak-resolution smoke failed')
        if 'SMOKE TEST PASSED' not in (OUT / f'{name}_smoke.log').read_text():
            raise RuntimeError('Missing successful smoke marker')
        with (OUT / f'{name}.log').open('a') as log:
            training_command = command + (['--evaluate-init'] if job['validation'] else [])
            process = spawn(training_command, log)
        status('training', pid=process.pid, command=training_command)
        if process.wait():
            raise RuntimeError('Training failed; inspect log')
        if not (output / 'COMPLETE').exists():
            raise RuntimeError('Training exited without COMPLETE')
        status('complete', pid=process.pid)
    except BaseException:
        status('failed', traceback=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=list(JOBS), required=True)
    main(parser.parse_args().job)
