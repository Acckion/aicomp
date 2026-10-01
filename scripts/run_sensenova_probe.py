"""Background controller: checksum-ready download, durable backup, local pilot.

Existing jobs are never stopped. All four devices must have enough headroom;
the worker enforces an 8.5 GiB PyTorch limit on each shared card.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/sensenova_probe'
RAM = Path('/dev/shm/aicomp_sensenova')
STORAGE = ROOT / 'checkpoints/gpu6_storage/sensenova'
GPUS = [3, 0, 1, 2]
PYTHON = '/home/fbohan/miniconda3/envs/AICOMP/bin/python'


def status(stage, **extra):
    value = {'stage': stage, 'time': time.time(), **extra}
    temp = OUT / 'controller_status.json.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(OUT / 'controller_status.json')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (RAM / 'download.complete').exists():
        pid = int((OUT / 'download.pid').read_text())
        if not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Model download stopped; see range_download.log')
        progress = json.loads((OUT / 'download_status.json').read_text()) if (OUT / 'download_status.json').exists() else {}
        status('waiting_for_verified_download', progress=progress)
        time.sleep(30)
    # Never write 30GB onto an accidentally unmounted local root directory.
    assert os.path.ismount(ROOT / 'checkpoints/gpu6_storage'), 'Remote storage is not mounted'
    assert shutil.disk_usage(STORAGE.parent).free > 40*1024**3
    STORAGE.mkdir(parents=True, exist_ok=True)
    status('saving_to_gpu6_storage', directory=str(STORAGE))
    for name in ['model', 'source', 'deps']:
        destination = STORAGE / name
        destination.mkdir(exist_ok=True)
        subprocess.run(['rsync', '-a', '--exclude=.cache', str(RAM/name)+'/', str(destination)+'/'], check=True)
    identity = json.loads((OUT / 'model_identity.json').read_text())
    # Verify the durable copy on its server, avoiding a second 30GB network read.
    expected = next(item['sha256'] for item in identity['files'] if item['name'] == 'ema.safetensors')
    command = ['ssh', '-i', '/home/fbohan/.ssh/aicomp_servers_ed25519', '-o', 'IdentitiesOnly=yes',
               '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'fbohan@222.20.97.104',
               'sha256sum /home2/fbohan/AIC_storage/sensenova/model/ema.safetensors']
    actual = subprocess.check_output(command, text=True).split()[0]
    assert actual == expected, 'Remote checkpoint checksum mismatch'
    (STORAGE / 'model_identity.json').write_text(json.dumps(identity, indent=2))
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
        if all(free[gpu] >= 9900 for gpu in GPUS):
            break
        status('waiting_for_gpu_headroom', gpus=GPUS, free_mib=free)
        time.sleep(30)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, GPUS)), OMP_NUM_THREADS='2',
               PYTHONPATH=str(STORAGE/'deps')+':'+str(STORAGE/'source'),
               LD_LIBRARY_PATH='/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib',
               HF_HOME=str(RAM/'hf'), TMPDIR=str(RAM), PYTHONUNBUFFERED='1')
    status('starting_worker', gpus=GPUS, source=str(STORAGE/'source'), model=str(STORAGE/'model'))
    with (OUT / 'worker.log').open('ab') as log:
        worker = subprocess.Popen([PYTHON, str(ROOT/'scripts/sensenova_probe.py'),
                                   '--source', str(STORAGE/'source'), '--model', str(STORAGE/'model')],
                                  stdout=log, stderr=subprocess.STDOUT, env=env)
        (OUT/'worker.pid').write_text(str(worker.pid))
        status('worker_running', pid=worker.pid, gpus=GPUS)
        returncode = worker.wait()
    if returncode:
        raise RuntimeError(f'Pilot worker exited {returncode}; see worker.log')
    status('complete', report=str(OUT/'summary.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        traceback.print_exc()
        sys.exit(1)
