"""Launch one isolated MAL or pixel-preserving dense-view screening run."""
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

ROOT=Path(__file__).resolve().parents[1]
ENV={**os.environ,'OMP_NUM_THREADS':'2',
     'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def active_workers(config):
    """Reject an existing worker even when its controller no longer exists."""
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            argv = (entry/'cmdline').read_bytes().decode().split('\0')
            if not any(Path(arg).name == 'train_mechanisms.py' for arg in argv):continue
            option = argv.index('--config')
            value = Path(argv[option+1])
            if not value.is_absolute():value = (entry/'cwd').resolve()/value
            if value.resolve() == config.resolve():found.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return found


def main(job):
    name,gpu = ('mal800',6) if job == 'mal' else ('dense_o2o800',4)
    directory=ROOT/'experiments/mechanisms'
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/(name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=directory/(name+'_status.json')
    child = None
    spawning = False
    pending_signal = None

    def stop_child():
        nonlocal child
        if child is None or child.poll() is not None:return
        # Every worker has its own process group. Only this controller's child
        # group is terminated, including any spawned data loader workers.
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=10)

    def on_signal(number, frame):
        nonlocal pending_signal
        if spawning:
            pending_signal=number
            return
        raise SystemExit(128+number)

    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)

    def spawn(command, env, log):
        nonlocal child, spawning
        # Do not allow TERM between process creation and storing the child PID:
        # defer the controller handler rather than block kernel signals, since
        # a blocked signal mask would be inherited by the detached worker.
        spawning=True
        try:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        finally:
            spawning=False
        if pending_signal is not None:
            raise SystemExit(128+pending_signal)
        return child
    def status(stage,**details):
        temp=state_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'stage':stage,'time':time.time(),'run':name,
                                   'gpu':gpu,'controller_pid':os.getpid(),**details},indent=2))
        temp.replace(state_path)
    try:
        config=ROOT/f'configs/{name}.yml'
        workers=active_workers(config)
        if workers:raise RuntimeError(f'Existing workers for {name}: {workers}')
        output=ROOT/'runs'/name
        if (output/'COMPLETE').exists():
            status('already_complete');return
        if (output/'metrics.jsonl').exists() or (output/'last.pth').exists():
            raise RuntimeError('Partial run requires explicit resume; refusing overwrite')
        status('waiting_for_memory',minimum_free_mib=9900)
        while True:
            query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                           '--format=csv,noheader,nounits'],text=True,timeout=10)
            free={int(a):int(b) for line in query.splitlines() for a,b in [line.split(',')]}
            if free[gpu]>=9900:break
            time.sleep(10)
        command=[sys.executable,str(ROOT/'scripts/train_mechanisms.py'),'--config',
                 str(ROOT/f'configs/{name}.yml'),'--init-checkpoint',
                 str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')]
        env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)}
        status('smoke')
        with (directory/(name+'_smoke.log')).open('a') as log:
            process=spawn(command+['--smoke'],env,log)
            code=process.wait()
            if code:raise RuntimeError(f'Smoke exit code {code}')
        with (directory/(name+'.log')).open('a') as log:
            process=spawn(command+['--evaluate-init'],env,log)
        status('training',pid=process.pid,log=str(directory/(name+'.log')),
               output=str(output))
        code=process.wait()
        if code:raise RuntimeError(f'Training exit code {code}')
        if not (output/'COMPLETE').exists():raise RuntimeError('No COMPLETE sentinel')
        status('complete',pid=process.pid)
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        stop_child()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',choices=['mal','dense'],required=True)
    main(parser.parse_args().job)
