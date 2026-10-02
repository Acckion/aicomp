"""Durable transfer supervision and final validation; never touches training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/gpu8_migration'
SSH = ['ssh', '-i', '/home/fbohan/.ssh/aicomp_servers_ed25519', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
HOST = 'fbohan@222.20.97.217'
TARGET = '/home/fbohan/AIC'
REMOTE_SHELL = 'ssh -i /home/fbohan/.ssh/aicomp_servers_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes'


def existing_transfer(token):
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if int(proc.name) == os.getpid() or proc.stat().st_uid != os.getuid():
                continue
            raw = (proc / 'cmdline').read_bytes().decode(errors='replace')
            if token in raw and HOST in raw and any(s in raw.split('\0')[0] for s in ['rsync', 'ssh', 'bash']):
                return True
        except (OSError, ValueError):
            pass
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'setup.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (OUT / 'input_manifest.json').exists():
        time.sleep(15)
    subprocess.run(['rsync', '-a', '-e', REMOTE_SHELL, str(OUT / 'input_manifest.json'), HOST + ':' + TARGET + '/migration/'], check=True)
    jobs = {
        'ENV_READY': ('AICOMP-env.tar.gz', 'set -e\nrsync -a --partial -e "' + REMOTE_SHELL + '" backups/AICOMP-env.tar.gz ' + HOST + ':' + TARGET + '/migration/\n' + REMOTE_SHELL + ' ' + HOST + " 'tar -xzf " + TARGET + "/migration/AICOMP-env.tar.gz -C /home/fbohan/miniconda3/envs/AICOMP && /home/fbohan/miniconda3/envs/AICOMP/bin/python /home/fbohan/miniconda3/envs/AICOMP/bin/conda-unpack && touch " + TARGET + "/migration/ENV_READY'"),
        'RGB_READY': ('data/train/', 'set -e\nrsync -a --partial -e "' + REMOTE_SHELL + '" data/train/ ' + HOST + ':' + TARGET + '/data/train/\n' + REMOTE_SHELL + ' ' + HOST + " 'touch " + TARGET + "/migration/RGB_READY'"),
        'IR_READY': ('AICOMP_IR_2000.zip', 'set -e\nrsync -a --partial -e "' + REMOTE_SHELL + '" /dev/shm/aicomp_gpu8_ironly.zip ' + HOST + ':' + TARGET + '/migration/AICOMP_IR_2000.zip\n' + REMOTE_SHELL + ' ' + HOST + " 'touch " + TARGET + "/migration/IR_READY'"),
        'WEIGHTS_READY': ('weights_epoch_', 'set -e\nrsync -a --partial -e "' + REMOTE_SHELL + '" runs/ft_aug800/weights_epoch_020.pth ' + HOST + ':' + TARGET + '/runs/ft_aug800/\nrsync -a --partial -e "' + REMOTE_SHELL + '" runs/ir_detector800/weights_epoch_024.pth ' + HOST + ':' + TARGET + '/runs/ir_detector800/\n' + REMOTE_SHELL + ' ' + HOST + " 'touch " + TARGET + "/migration/WEIGHTS_READY'"),
    }
    children = {}
    while True:
        try:
            result = subprocess.run(SSH + [HOST, 'ls ' + TARGET + '/migration/*_READY'], text=True, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            time.sleep(30)
            continue
        ready = {Path(s).name for s in result.stdout.splitlines()}
        for marker, (token, command) in jobs.items():
            if marker in ready:
                continue
            child = children.get(marker)
            if child is not None and child.poll() is None:
                continue
            if existing_transfer(token):
                continue
            log = (OUT / (marker + '_transfer.log')).open('ab')
            children[marker] = subprocess.Popen(['bash', '-c', command], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        (OUT / 'setup_status.json').write_text(json.dumps({'stage': 'copying', 'ready': sorted(ready), 'pending': sorted(set(jobs) - ready), 'time': time.time()}, indent=2))
        if set(jobs) <= ready:
            break
        time.sleep(30)
    command = ['env', 'CUDA_VISIBLE_DEVICES=4', 'LD_LIBRARY_PATH=/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib',
               '/home/fbohan/miniconda3/envs/AICOMP/bin/python', TARGET + '/scripts/validate_gpu8_inputs.py']
    with (OUT / 'validation.log').open('ab') as stream:
        result = subprocess.run(SSH + [HOST, *command], stdout=stream, stderr=stream)
    stage = 'complete' if result.returncode == 0 else 'validation_failed'
    (OUT / 'setup_status.json').write_text(json.dumps({'stage': stage, 'time': time.time(), 'exit_code': result.returncode}, indent=2))
    if result.returncode:
        raise RuntimeError('Remote inputs/runtime validation failed; training remains blocked')


if __name__ == '__main__':
    main()
