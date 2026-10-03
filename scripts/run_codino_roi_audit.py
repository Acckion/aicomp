"""Restore existing isolated runtime and audit individual heads after owned copies finish."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RAM = Path('/dev/shm/aicomp_codino')
OUT = ROOT/'experiments/codino_roi_probe'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    owner = (OUT/'controller.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'validation/COMPLETE').exists(), 'Do not overwrite a completed audit'
    child = None
    def status(stage, **extra):
        p = OUT/'status.tmp'
        p.write_text(json.dumps({'stage':stage, 'controller_pid':os.getpid(), 'time':time.time(), **extra}, indent=2))
        p.replace(OUT/'status.json')
    signal.signal(signal.SIGTERM, lambda n,f: sys.exit(128+n))
    signal.signal(signal.SIGINT, lambda n,f: sys.exit(128+n))
    def run(command, filename, env=None):
        nonlocal child
        with (OUT/filename).open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=env)
            status('executing', task=filename, worker_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError(f'{filename} failed with{code}; no blind retry')
    try:
        copies = [
            (RAM/'runtime_py311_torch20_cu118.tar', 5597562880,
             'f3cc04121eb07870b82b917b92f0a492b0fe237c298271824fd31532ebcf0302', 3482430),
            (RAM/'codino1600_best.pth', 1714596763,
             'ac310a72be19de1afb0871e9fbcf23415a87293534b668465d70c2bed5804e7b', 3483832),
        ]
        for path, size, sha, pid in copies:
            while not path.exists() or path.stat().st_size != size:
                process = Path(f'/proc/{pid}/cmdline')
                assert process.exists() and 'scp' in process.read_text() and path.name in process.read_text(), 'Copy ended without complete verified artifact'
                status('waiting_owned_copy', copy_pid=pid, artifact=path.name,
                    bytes=path.stat().st_size if path.exists() else 0, expected_bytes=size)
                time.sleep(15)
            assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == sha
        (OUT/'verified_artifacts.json').write_text(json.dumps([
            {'path':str(p), 'bytes':s, 'sha256':h} for p,s,h,_ in copies], indent=2)+'\n')
        runtime = RAM/'venv/bin/python'
        if not runtime.exists():
            run(['tar','-xf',str(copies[0][0]),'-C',str(RAM)], 'restore.log')
        env = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
            'CODINO_VENDOR':str(RAM/'vendor/Co-DETR'), 'LD_LIBRARY_PATH':'', 'TMPDIR':str(RAM/'tmp')}
        (RAM/'tmp').mkdir(exist_ok=True)
        run([str(runtime),'-c', 'import torch,numpy,mmcv,mmcv.ops; assert torch.__version__=="2.0.0+cu118"; assert numpy.__version__=="1.26.4"; assert mmcv.__version__=="1.7.2"; print("isolated runtime verified")'], 'runtime_verify.log', env)
        gpu = (ROOT/'experiments/mechanism_trials/gpu4.lock').open('a')
        while True:
            try:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                free = int(subprocess.check_output(['nvidia-smi','-i','4',
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
                if free >= int(3.4*1024)+512:
                    break
                fcntl.flock(gpu, fcntl.LOCK_UN)
                status('waiting_memory', free_mib=free)
            except BlockingIOError:
                status('waiting_owned_gpu_lock')
            time.sleep(15)
        env['CUDA_VISIBLE_DEVICES'] = '4'
        command = [str(runtime),'-u',str(ROOT/'scripts/audit_codino_roi.py'),
            '--checkpoint',str(copies[1][0]),'--memory-cap-gib','3.4']
        run(command+['--output',str(OUT/'smoke'),'--smoke'], 'smoke.log', env)
        smoke = json.loads((OUT/'smoke/smoke.json').read_text())
        assert all(r['images']==2 and r['finite'] for r in smoke['results'].values())
        run(command+['--output',str(OUT/'validation')], 'validation.log', env)
        assert (OUT/'validation/COMPLETE').exists()
        status('complete', report=str(OUT/'validation/report.json'))
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


if __name__ == '__main__':
    main()
