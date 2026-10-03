"""Launch one preflighted full-frame arm within a shared-GPU memory budget."""
import argparse
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

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',choices=['scene_whole_head_real','scene_whole_head_sham'],required=True)
    parser.add_argument('--gpu-index',type=int,required=True)
    parser.add_argument('--output-target',type=Path,required=True)
    args=parser.parse_args()
    from src.core.yaml_utils import load_config
    config_path=ROOT/'configs'/f'{args.name}.yml'
    config=load_config(str(config_path),{})
    out=ROOT/'experiments/whole_frame_head'/args.name;out.mkdir(parents=True,exist_ok=True)
    preflight=json.loads((out/'preflight.json').read_text())
    assert preflight['stage']=='passed' and preflight['name']==args.name
    assert preflight['frozen_visual_state_exact'] and preflight['strict_reload_exact']
    assert preflight['sham_reconstruction_identity_exact']
    cap=float(config['gpu_memory_limit_gib']);assert preflight['memory_cap_gib']<=cap<=10
    owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    child=None
    def status(stage,**extra):
        p=out/'status.tmp';p.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),
            'gpu':args.gpu_index,'name':args.name,'memory_cap_gib':cap,'time':time.time(),**extra},indent=2));p.replace(out/'status.json')
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n))
    signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    try:
        while True:
            try:
                fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=int(cap*1024)+512:break
                fcntl.flock(gpu,fcntl.LOCK_UN);status('waiting_memory',free_mib=free)
            except BlockingIOError:status('waiting_owned_gpu_lock')
            time.sleep(15)
        train_path=Path(config['train_dataloader']['dataset']['ann_file'])
        val_path=Path(config['val_dataloader']['dataset']['ann_file'])
        train=json.loads(train_path.read_text());val=json.loads(val_path.read_text())
        assert len(train['images'])==1610 and len(val['images'])==390
        assert {i['id'] for i in train['images']}.isdisjoint(i['id'] for i in val['images'])
        run=Path(config['output_dir']);target=args.output_target
        target.mkdir(parents=True,exist_ok=True)
        if run.exists() or run.is_symlink():assert run.resolve()==target.resolve()
        else:run.symlink_to(target,target_is_directory=True)
        assert not (run/'metrics.jsonl').exists() and not (run/'COMPLETE').exists(),'Explicit recovery needed, no duplicate training'
        checkpoint=ROOT/'runs/scene_obj365_pool800/weights_epoch_020.pth'
        manifest={'parent_checkpoint':str(checkpoint),'parent_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
            'train_annotation_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
            'val_annotation_sha256':hashlib.sha256(val_path.read_bytes()).hexdigest(),
            'geometry':[1088,1920],'mode':config['WholeFrameHeadDFINE']['mode'],
            'frozen':'backbone and encoder; decoder only trainable','epochs':8,'warmup_epochs':1,
            'batch':1,'accumulation':8,'base_lr':config['optimizer']['lr'],'preflight':preflight,
            'comparison':'Same original-image augmentation, parent, geometry, schedule and frozen visual features. Sham reconstructs native canvas from800-square RGB.',
            'limitations':'Only official train pixels and labels. Low-resolution originals gain no new pixels by upscaling. No phase2 fitting, voting or online improvement claim.'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        command=[sys.executable,'-u',str(ROOT/'scripts/train_ir_content.py'),'--config',str(config_path),
                 '--init-checkpoint',str(checkpoint),'--evaluate-init']
        with (out/'training.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu_index),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            status('training',worker_pid=child.pid);code=child.wait()
        if code:raise RuntimeError(f'Worker failed with{code}; no blind retry')
        assert (run/'COMPLETE').exists();status('complete')
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            except ProcessLookupError:pass


if __name__=='__main__':
    sys.path[:0]=[str(ROOT/'D-FINE'),str(ROOT/'scripts')]
    main()
