"""Exercise controller TERM cleanup with isolated dummy workers, without CUDA."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def live(pid):
    try:
        # Reaped by the host later: zombies consume no CPU/CUDA and are stopped.
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:return False


def main():
    source=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='check-dense-launcher-') as temporary:
        root=Path(temporary)
        (root/'scripts').mkdir();(root/'configs').mkdir();(root/'bin').mkdir()
        (root/'configs/mal800.yml').write_text('{}')
        fake_smi=root/'bin/nvidia-smi'
        fake_smi.write_text('#!/bin/sh\necho "6, 24110"\n')
        fake_smi.chmod(0o755)
        (root/'scripts/train_mechanisms.py').write_text(
            'import os,subprocess,sys,time\nfrom pathlib import Path\n'
            'if "--smoke" in sys.argv:sys.exit(0)\n'
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(300)"])\n'
            'Path(__file__).resolve().parents[1].joinpath("worker_pids").write_text(f"{os.getpid()} {p.pid}")\n'
            'time.sleep(300)\n')
        entry=('import sys;from pathlib import Path;'
               f'sys.path.insert(0,{str(source)!r});'
               'import run_dense_supervision as m;'
               f'm.ROOT=Path({str(root)!r});'
               'm.main("mal")')
        env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH']}
        log=(root/'controller.log').open('w')
        controller=subprocess.Popen([sys.executable,'-c',entry],env=env,stdout=log,
                                    stderr=subprocess.STDOUT,start_new_session=True)
        pids=[]
        try:
            deadline=time.monotonic()+15
            while not (root/'worker_pids').exists() and time.monotonic()<deadline:
                if controller.poll() is not None:raise RuntimeError('Dummy controller ended early')
                time.sleep(.05)
            pids=list(map(int,(root/'worker_pids').read_text().split()))
            sys.path.insert(0,str(source))
            from run_dense_supervision import active_workers
            assert pids[0] in active_workers(root/'configs/mal800.yml')
            controller.send_signal(signal.SIGTERM)
            controller.wait(timeout=20)
            deadline=time.monotonic()+5
            while any(live(pid) for pid in pids) and time.monotonic()<deadline:time.sleep(.05)
            assert not any(live(pid) for pid in pids), 'Worker process group survived controller TERM'
            print('CONTROLLER TERM CHECK PASSED: owned worker and child stopped; /proc active-worker detection passed')
        finally:
            for pid in [controller.pid,*pids]:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
            log.close()


if __name__ == '__main__':main()
