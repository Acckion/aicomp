"""One detached per-card controller for a native-grid factorial experiment."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=['native_grid1','native_grid1_control','native_grid2','native_grid2_control'], required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    out = ROOT / 'experiments/native_grid' / args.name
    out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(args.gpu_index),
           'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'PYTHONUNBUFFERED':'1',
           'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
    def status(stage, **details):
        p = out / 'status.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage':stage, 'gpu':args.gpu_index, 'name':args.name,
                                  'controller_pid':os.getpid(), 'time':time.time(), **details}, indent=2))
        tmp.replace(p)
    def run(stage, command):
        nonlocal child
        while True:
            free = int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),
                       '--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
            if free >= 9216:
                break
            status('waiting_memory', next_stage=stage, free_mib=free)
            time.sleep(15)
        with (out / f'{stage}.log').open('ab') as stream:
            child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(stage, worker_pid=child.pid)
        if child.wait():
            raise RuntimeError(f'{stage} failed; inspect {out / (stage + ".log")}')
    signal.signal(signal.SIGTERM, lambda n,f:sys.exit(128+n))
    signal.signal(signal.SIGINT, lambda n,f:sys.exit(128+n))
    try:
        assert subprocess.check_output(['findmnt','-n','-T',str(ROOT/'checkpoints/gpu6_storage'),'-o','FSTYPE'], text=True).strip() == 'fuse.sshfs'
        target = ROOT / 'checkpoints/gpu6_storage/native_grid/runs' / args.name
        target.mkdir(parents=True, exist_ok=True)
        output = ROOT / 'runs' / args.name
        if output.is_symlink():
            assert output.resolve() == target.resolve()
        elif output.exists():
            raise RuntimeError('Unexpected existing output directory')
        else:
            output.symlink_to(target, target_is_directory=True)
        assert not (output / 'COMPLETE').exists()
        if not (out / 'preflight.json').exists():
            run('preflight', [str(ROOT/'scripts/native_grid_preflight.py'),'--name',args.name])
        assert json.loads((out/'preflight.json').read_text())['stage'] == 'passed'
        command = [str(ROOT/'scripts/train_native_grid.py'),'--config',str(ROOT/'configs'/f'{args.name}.yml'),
                   '--init-checkpoint',str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')]
        run('smoke', command + ['--smoke'])
        run('training', command + ['--evaluate-init'])
        assert (output / 'COMPLETE').exists()
        status('complete')
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
                child.wait()
            except ProcessLookupError:
                pass
        lock.close()


if __name__ == '__main__':
    main()
