"""Audit two immutable whole-frame checkpoints on the complete heldout fold."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--gpu-index',type=int,default=6);args=parser.parse_args()
    out=ROOT/f'experiments/whole_frame_val_e{args.epoch}';out.mkdir(parents=True,exist_ok=True)
    owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'COMPLETE').exists(),'Existing completed validation must not be overwritten'
    child=None
    def status(stage,**extra):
        p=out/'status.tmp';p.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),
            'epoch':args.epoch,'time':time.time(),**extra},indent=2));p.replace(out/'status.json')
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    try:
        gpu=(ROOT/f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
        while True:
            try:
                fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=2560:break
                fcntl.flock(gpu,fcntl.LOCK_UN);status('waiting_memory',free_mib=free)
            except BlockingIOError:status('waiting_owned_gpu_lock')
            time.sleep(15)
        reports={}
        for mode in ['real','sham']:
            name='scene_whole_head_'+mode
            command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_variants.py'),
                '--config',str(ROOT/'configs'/f'{name}.yml'),
                '--checkpoint',str(ROOT/'runs'/name/f'weights_epoch_{args.epoch:03d}.pth'),
                '--annotations',str(ROOT/'data/annotations/scene_val.json'),'--image-root',str(ROOT/'data/train'),
                '--size','800','--amp','--native-top100','--require-ema','--single-method',
                '--gpu-memory-limit-gib','2','--output',str(out/mode)]
            with (out/f'{mode}.log').open('ab') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu_index),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
                status('evaluating',mode=mode,worker_pid=child.pid);code=child.wait()
            if code:raise RuntimeError(f'Validation{mode} failed with{code}; no blind retry')
            reports[mode]=json.loads((out/mode/'results.json').read_text())
            assert reports[mode]['images']==390 and not reports[mode]['limited_subset']
            identity=json.loads((out/mode/'cache_identity.json').read_text())
            assert identity['whole_frame_geometry']==[1088,1920]
            assert identity['whole_frame_pixel_source']==mode
        (out/'report.json').write_text(json.dumps({'epoch':args.epoch,'results':reports,
            'scope':'Complete grouped390, fixed EMA checkpoints; no test predictions or training changes.'},indent=2)+'\n')
        (out/'COMPLETE').write_text('ok\n');status('complete')
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            except ProcessLookupError:pass


if __name__=='__main__':main()
