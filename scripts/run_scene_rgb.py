"""Memory-bounded grouped RGB training on a dedicated remote project copy."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/scene_rgb_gpu6'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    lock = (OUT / 'gpu2.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
           'LD_LIBRARY_PATH':str(Path(sys.prefix)/'lib/python3.11/site-packages/nvidia/nvjitlink/lib')}
    def status(stage, **extra):
        tmp = OUT / 'status.tmp'
        tmp.write_text(json.dumps({'stage':stage,'time':time.time(),'controller_pid':os.getpid(),**extra},indent=2))
        tmp.replace(OUT / 'status.json')
    def run(stage, command):
        nonlocal child
        while True:
            free = int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free >= 7168:
                break
            status('waiting_memory', next_stage=stage, free_mib=free)
            time.sleep(15)
        with (OUT / f'{stage}.log').open('ab') as log:
            child = subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage, worker_pid=child.pid)
        if child.wait():
            raise RuntimeError(f'{stage} failed, inspect its log')
    signal.signal(signal.SIGTERM, lambda n,f:sys.exit(128+n))
    signal.signal(signal.SIGINT, lambda n,f:sys.exit(128+n))
    try:
        assert not (ROOT / 'runs/scene_rgb800/COMPLETE').exists()
        run('preflight', [str(ROOT/'scripts/scene_rgb_preflight.py')])
        assert json.loads((OUT/'preflight.json').read_text())['actual_updates'] == 3
        command = [str(ROOT/'scripts/train_ir_content.py'),'--config',str(ROOT/'configs/scene_rgb800.yml')]
        run('smoke', command + ['--smoke'])
        run('training', command)
        assert (ROOT/'runs/scene_rgb800/COMPLETE').exists()
        status('complete')
    except Exception as error:
        status('failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            child.wait()


if __name__ == '__main__':
    main()
