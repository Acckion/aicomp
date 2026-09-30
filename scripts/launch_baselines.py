"""Wait for a complete training upload, validate it, smoke-test and train on 8 GPUs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/fbohan/miniconda3/envs/AICOMP/bin/python')
ARCHIVE = ROOT / '初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip'
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)
ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
           LD_LIBRARY_PATH='/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib')


def status(stage, **kwargs):
    record = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs}
    temp = LOGS / 'launcher_status.tmp'
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(LOGS / 'launcher_status.json')
    print(json.dumps(record), flush=True)


def wait_ready():
    previous_size = None
    while True:
        size = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
        env_ready = (LOGS / 'ENV_READY').exists()
        weight_ready = (ROOT / 'checkpoints/READY').exists()
        data_ready = size > 0 and size == previous_size and zipfile.is_zipfile(ARCHIVE)
        if env_ready and weight_ready and data_ready:
            return
        status('waiting', environment_ready=env_ready, weights_ready=weight_ready,
               training_archive_bytes=size, training_archive_complete=data_ready)
        previous_size = size
        time.sleep(30)


def main():
    lock = (LOGS / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_ready()
    status('preparing_data')
    with (LOGS / 'prepare_data.log').open('a') as log:
        subprocess.run([str(PYTHON), str(ROOT / 'scripts/prepare_data.py')], cwd=ROOT,
                       env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Respect other GPU work if the machine changed while the upload was pending.
    while True:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if not processes:
            break
        status('waiting_for_free_gpus', compute_pids=processes)
        time.sleep(30)
    specs = [('rgb1600', '0,1,2,3', '29601'), ('rgb2000', '4,5,6,7', '29602')]
    # Test both modes, distributed communication, backward, AMP and validation,
    # before either long-running experiment starts.
    for name, devices, port in specs:
        status('smoke_test', experiment=name)
        smoke_config = ROOT / f'configs/{name}_smoke.yml'
        smoke_config.write_text(f"__include__: ['./{name}.yml']\noutput_dir: {ROOT}/runs/smoke_{name}\n")
        command = [str(PYTHON), '-m', 'torch.distributed.run', '--nproc_per_node=4',
                   f'--master_port={port}', str(ROOT / 'scripts/train_baseline.py'),
                   '--config', str(smoke_config), '--smoke']
        with (LOGS / f'{name}_smoke.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=devices),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    jobs = []
    for name, devices, port in specs:
        if (ROOT / f'runs/{name}/COMPLETE').exists():
            continue
        command = [str(PYTHON), '-m', 'torch.distributed.run', '--nproc_per_node=4',
                   f'--master_port={port}', str(ROOT / 'scripts/train_baseline.py'),
                   '--config', str(ROOT / f'configs/{name}.yml')]
        last = ROOT / f'runs/{name}/last.pth'
        if last.exists():
            command.extend(['--resume', str(last)])
        log = (LOGS / f'{name}.log').open('a')
        process = subprocess.Popen(command, cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=devices),
                                   stdout=log, stderr=subprocess.STDOUT)
        jobs.append((name,process,log))
    while any(p.poll() is None for _,p,_ in jobs):
        status('training', jobs={name:{'pid':p.pid,'exit_code':p.poll()} for name,p,_ in jobs})
        time.sleep(30)
    codes = {name:p.returncode for name,p,_ in jobs}
    for _,_,log in jobs:
        log.close()
    if any(code != 0 for code in codes.values()):
        raise RuntimeError(f'Training failed: {codes}; inspect experiment logs')
    if not all((ROOT / f'runs/{name}/COMPLETE').exists() for name,_,_ in specs):
        raise RuntimeError('Training exited without completion markers')
    status('complete', exit_codes=codes)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
