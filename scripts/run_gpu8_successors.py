"""GPU8-local successor queues; prerequisites are published from GPU7."""
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
PYTHON = '/home/fbohan/miniconda3/envs/AICOMP/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', choices=['ir', 'roi'], required=True)
    args = parser.parse_args()
    ir = args.direction == 'ir'
    gpu = 4 if ir else 2
    names = ['ir_reliability800', 'ir_reliability_control800'] if ir else ['roi_coverage_control800']
    out = ROOT / 'experiments' / ('next_' + args.direction)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'gpu8_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    device_lock = None
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
           'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}

    def status(stage, **fields):
        temp = out / 'status.gpu8.tmp'
        temp.write_text(json.dumps({'stage': stage, 'server': 'GPU8', 'host': '222.20.97.217', 'gpu': gpu,
                                    'controller_pid': os.getpid(), 'queued_runs': names, 'time': time.time(), **fields}, indent=2))
        temp.replace(out / 'status.json')

    def run(command, log, stage, **fields):
        nonlocal child
        with (out / log).open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(stage, worker_pid=child.pid, **fields)
        if child.wait():
            raise RuntimeError(f'{stage} failed; see {log}')

    signal.signal(signal.SIGTERM, lambda n, f: sys.exit(128 + n))
    signal.signal(signal.SIGINT, lambda n, f: sys.exit(128 + n))
    try:
        markers = ['ENV_READY', 'RGB_READY', 'WEIGHTS_READY', 'VALIDATED'] + (['IR_READY'] if ir else [])
        while True:
            missing = [m for m in markers if not (ROOT / 'migration' / m).exists()]
            if not missing:
                break
            status('waiting_for_remote_setup', missing=missing); time.sleep(15)
        ready = ROOT / 'migration/upstream' / (args.direction + '.ready.json')
        while not ready.exists():
            status('waiting_for_gpu7_prerequisites'); time.sleep(15)
        upstream = json.loads(ready.read_text())
        assert upstream['stage'] == 'complete' and upstream['direction'] == args.direction
        device_lock = (ROOT / 'experiments/mechanism_trials' / f'gpu{gpu}.lock').open('a')
        while True:
            try:
                fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                status('waiting_for_gpu_lock'); time.sleep(15)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 9216:
                break
            status('waiting_for_memory', free_mib=free); time.sleep(15)
        for name in names:
            (ROOT / 'runs' / name).mkdir(parents=True, exist_ok=True)
        run([PYTHON, str(ROOT / 'scripts' / f'next_{args.direction}_preflight.py')], 'gpu8_preflight.log', 'preflight')
        assert json.loads((out / 'preflight.json').read_text())['stage'] == 'passed'
        for name in names:
            output = ROOT / 'runs' / name
            if (output / 'COMPLETE').exists():
                continue
            if (output / 'metrics.jsonl').exists():
                raise RuntimeError(f'Partial run requires explicit recovery: {name}')
            worker = ROOT / 'scripts' / ('train_ir_content.py' if ir else 'train_native_roi.py')
            command = [PYTHON, str(worker), '--config', str(ROOT / 'configs' / f'{name}.yml'),
                       '--init-checkpoint', str(ROOT / 'runs/ft_aug800/weights_epoch_020.pth')]
            run(command + ['--smoke'], name + '_gpu8_smoke.log', 'smoke', run=name)
            run(command + ['--evaluate-init'], name + '_gpu8.log', 'training', run=name, epochs=8, memory_limit_gib=8.5)
            assert (output / 'COMPLETE').exists()
            if ir and name == names[0]:
                run([PYTHON, str(ROOT / 'scripts/next_ir_ablations.py'), '--checkpoint', str(output / 'best.pth')],
                    'gpu8_ablations.log', 'ablations', run=name)
        status('complete')
    except BaseException:
        status('failed', traceback=traceback.format_exc()); raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            except ProcessLookupError:
                pass
        if device_lock is not None:
            device_lock.close()


if __name__ == '__main__':
    main()
