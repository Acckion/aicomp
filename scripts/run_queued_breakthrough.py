"""Wait for existing matched experiments, then preflight and train successors."""
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
PLANS = {
    'ir': (6, 'ir_content', ['ir_content800', 'ir_content_control800'], ['ir_reliability800', 'ir_reliability_control800']),
    'roi': (1, 'native_roi', ['native_roi800', 'native_roi_control800'], ['roi_coverage800', 'roi_coverage_control800']),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', choices=PLANS, required=True)
    direction = parser.parse_args().direction
    gpu, prerequisite, previous, runs = PLANS[direction]
    out = ROOT / 'experiments' / ('next_' + direction)
    out.mkdir(parents=True, exist_ok=True)
    controller = (out / 'controller.lock').open('a')
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    gpu_lock = None
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
           'LD_LIBRARY_PATH': '/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}

    def status(stage, **fields):
        temp = out / 'status.tmp'
        temp.write_text(json.dumps({'stage': stage, 'gpu': gpu, 'controller_pid': os.getpid(), 'time': time.time(),
                                    'prerequisite': previous, 'queued_runs': runs, **fields}, indent=2))
        temp.replace(out / 'status.json')

    def run(command, log, stage, **fields):
        nonlocal child
        with (out / log).open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(stage, worker_pid=child.pid, **fields)
        if child.wait():
            raise RuntimeError(f'{stage} failed: {log}; queue stops without changing existing tasks')

    signal.signal(signal.SIGTERM, lambda n, f: sys.exit(128 + n))
    signal.signal(signal.SIGINT, lambda n, f: sys.exit(128 + n))
    try:
        while True:
            try:
                state = json.loads((ROOT / 'experiments' / prerequisite / 'status.json').read_text())
            except (OSError, ValueError):
                state = {}
            # Status alone is insufficient: require both actual completed runs.
            if state.get('stage') == 'complete' and all((ROOT / 'runs' / name / 'COMPLETE').exists() for name in previous):
                break
            status('waiting_for_current_experiments', current_stage=state.get('stage'), current_run=state.get('job', state.get('run')))
            time.sleep(15)
        gpu_lock = (ROOT / 'experiments/mechanism_trials' / f'gpu{gpu}.lock').open('a')
        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status('waiting_for_gpu_lock'); time.sleep(15)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 9216:
                break
            status('waiting_for_memory', free_mib=free, required_free_mib=9216); time.sleep(15)
        assert subprocess.check_output(['findmnt', '-n', '-T', str(ROOT / 'checkpoints/gpu6_storage'), '-o', 'FSTYPE'], text=True).strip() == 'fuse.sshfs'
        for name in runs:
            destination = ROOT / 'checkpoints/gpu6_storage' / ('next_' + direction) / 'runs' / name
            destination.mkdir(parents=True, exist_ok=True)
            local = ROOT / 'runs' / name
            if local.is_symlink():
                assert local.resolve() == destination.resolve()
            elif local.exists():
                raise RuntimeError(f'Refusing existing unrelated output: {local}')
            else:
                local.symlink_to(destination, target_is_directory=True)
        run([sys.executable, str(ROOT / 'scripts' / f'next_{direction}_preflight.py')], 'preflight.log', 'preflight')
        assert json.loads((out / 'preflight.json').read_text())['stage'] == 'passed'
        for name in runs:
            output = ROOT / 'runs' / name
            if (output / 'COMPLETE').exists():
                continue
            if (output / 'metrics.jsonl').exists():
                raise RuntimeError(f'Partial inference-only run requires explicit recovery: {name}')
            command = [sys.executable, str(ROOT / 'scripts/train_ir_content.py' if direction == 'ir' else ROOT / 'scripts/train_native_roi.py'),
                       '--config', str(ROOT / 'configs' / f'{name}.yml'), '--init-checkpoint', str(ROOT / 'runs/ft_aug800/weights_epoch_020.pth')]
            run(command + ['--smoke'], name + '_smoke.log', 'smoke', run=name)
            run(command + ['--evaluate-init'], name + '.log', 'training', run=name, epochs=8)
            assert (output / 'COMPLETE').exists()
            if direction == 'ir' and name == runs[0]:
                run([sys.executable, str(ROOT / 'scripts/next_ir_ablations.py'), '--checkpoint', str(output / 'best.pth')], 'ablations.log', 'ablations', run=name)
        status('complete')
    except BaseException:
        status('failed', traceback=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            except ProcessLookupError:
                pass
        if gpu_lock is not None:
            gpu_lock.close()


if __name__ == '__main__':
    main()
