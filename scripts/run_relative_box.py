"""Detached, locked launcher for the single localization mechanism experiment."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/relative_box800'
LOGS = ROOT / 'experiments/mechanism_trials'
GPU = 5
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU), 'OMP_NUM_THREADS': '2',
       'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (LOGS / 'relative_box800.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **details):
        path = LOGS / 'relative_box800_status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage': stage, 'time': time.time(),
                                    'controller_pid': os.getpid(), 'gpu': GPU,
                                    'run_dir': str(RUN), **details}, indent=2))
        tmp.replace(path)
    try:
        if any((RUN / name).exists() for name in ('COMPLETE', 'metrics.jsonl', 'last.pth')):
            raise RuntimeError('Refusing to overwrite an existing or partial training run')
        status('waiting_for_memory')
        while True:
            free = int(subprocess.check_output(
                ['nvidia-smi', f'--id={GPU}', '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 9200:
                break
            time.sleep(10)
        command = [sys.executable, str(ROOT / 'scripts/train_mechanisms.py'),
                   '--config', str(ROOT / 'configs/relative_box800.yml'),
                   '--init-checkpoint', str(ROOT / 'runs/ft_aug800/weights_epoch_020.pth')]
        smoke_log = LOGS / 'relative_box800_smoke.log'
        # A successful exact-config smoke may have been run during preparation.
        if not smoke_log.exists() or 'SMOKE TEST PASSED' not in smoke_log.read_text():
            status('smoke')
            with smoke_log.open('a') as log:
                subprocess.run(command + ['--smoke'], cwd=ROOT, env=ENV,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        with (LOGS / 'relative_box800.log').open('a') as log:
            process = subprocess.Popen(command + ['--evaluate-init'], cwd=ROOT, env=ENV,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (LOGS / 'relative_box800_train.pid').write_text(str(process.pid) + '\n')
        status('training', pid=process.pid, log=str(LOGS / 'relative_box800.log'))
        returncode = process.wait()
        if returncode != 0 or not (RUN / 'COMPLETE').exists():
            raise RuntimeError(f'Training exited with {returncode}; COMPLETE={(RUN / "COMPLETE").exists()}')
        status('complete', pid=process.pid)
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
