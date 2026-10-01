"""Wait for training-only teacher cache, then smoke and start one matched trial."""
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

from vl_distill_cache import write_json,sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/vl_distillation'
JOBS={'vl_control800':4,'vl_distill800':7}
ENV={**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false',
     'LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def main(job):
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/f'{job}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=JOBS[job]
    gpu_lock=(ROOT/f'experiments/mechanism_trials/gpu{gpu}.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    def status(stage,**details):
        write_json(OUT/f'{job}_status.json',{'stage':stage,'job':job,'gpu':gpu,'time':time.time(),
                   'controller_pid':os.getpid(),**details})
    def stopped(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    def spawn(command,log):
        nonlocal child
        child=subprocess.Popen(command,cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)},
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return child
    try:
        output=ROOT/'runs'/job
        if (output/'COMPLETE').exists() or (output/'metrics.jsonl').exists():
            raise RuntimeError('Refusing to overwrite existing trial')
        while not (OUT/'teacher/COMPLETE').exists() or not (OUT/'preflight.json').exists():
            status('waiting_for_teacher_and_preflight');time.sleep(15)
        preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['stage']=='passed'
        # Root space is shared and limited. Full resumable states go to the
        # already-authorized GPU6 storage mount, never its full /home partition.
        mount=ROOT/'checkpoints/gpu6_storage'
        filesystem=subprocess.check_output(['findmnt','-n','-T',str(mount),'-o','FSTYPE'],text=True).strip()
        if filesystem!='fuse.sshfs':raise RuntimeError('GPU6 storage must be mounted before training')
        remote=mount/'vl_distillation/runs'/job;remote.mkdir(parents=True,exist_ok=True)
        if output.is_symlink():assert output.resolve()==remote.resolve()
        elif output.exists():raise RuntimeError('Existing nonsymlink run directory; will not replace')
        else:output.symlink_to(remote,target_is_directory=True)
        parent=ROOT/'runs/ft_aug800/weights_epoch_020.pth'
        assert sha256(parent)==preflight['parent_sha256']
        assert sha256(OUT/'teacher/targets.npz')==preflight['teacher_targets_sha256']
        metadata={'job':job,'epochs':8,'train_images':1600,'validation_images':400,
                  'gpu':gpu,'parent_sha256':preflight['parent_sha256'],
                  'teacher_targets_sha256':preflight['teacher_targets_sha256'],
                  'feature_weight':0 if job=='vl_control800' else .1,
                  'semantic_weight':0 if job=='vl_control800' else .02,
                  'inference':'native D-FINE, training-only projector removed',
                  'checkpoint_storage':'GPU6 SSHFS; full last/best include optimizer, scaler and RNG',
                  'interpretation':'Local GT-matched distillation hypothesis, not full DK-DETR reproduction.'}
        write_json(OUT/f'{job}_identity.json',metadata)
        command=[sys.executable,str(ROOT/'scripts/train_mechanisms.py'),'--config',str(ROOT/f'configs/{job}.yml'),
                 '--init-checkpoint',str(parent)]
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=9900:break
            status('waiting_for_memory',free_mib=free,minimum_free_mib=9900);time.sleep(15)
        status('smoke')
        with (OUT/f'{job}_smoke.log').open('ab') as log:
            if spawn(command+['--smoke'],log).wait():raise RuntimeError('Peak-resolution smoke failed')
        assert 'SMOKE TEST PASSED' in (OUT/f'{job}_smoke.log').read_text()
        with (OUT/f'{job}.log').open('ab') as log:
            spawn(command+['--evaluate-init'],log)
        status('training',pid=child.pid,command=command+['--evaluate-init'])
        if child.wait():raise RuntimeError('Training failed; inspect trial log')
        assert (output/'COMPLETE').exists()
        status('complete',pid=child.pid)
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            except ProcessLookupError:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',choices=list(JOBS),required=True)
    main(parser.parse_args().job)
