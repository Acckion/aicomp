"""Detached transfer, relocation and launch of a fresh scene-group baseline."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/scene_rgb_gpu6_setup'
OUT.mkdir(parents=True, exist_ok=True)
HOST = 'fbohan@222.20.97.104'
SSH = ['ssh','-i',str(Path.home()/'.ssh/aicomp_servers_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=10',HOST]
REMOTE = '/home2/fbohan/AIC'
REMOTE_ENV = '/home2/fbohan/miniconda3/envs/AICOMP'


def main():
    lock = (OUT/'setup.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage, **extra):
        p = OUT/'status.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage':stage,'time':time.time(),'pid':os.getpid(),**extra},indent=2));tmp.replace(p)
    def run(stage, args):
        status(stage)
        print(stage, flush=True)
        subprocess.run(args, check=True, cwd=ROOT)
    try:
        run('prepare_remote',SSH+['mkdir -p /home2/fbohan/AIC/{backups,checkpoints,experiments/scene_groups,data/train} /home2/fbohan/miniconda3/envs'])
        archive = OUT/'source.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for folder in ['D-FINE','scripts','configs']:
                for path in (ROOT/folder).rglob('*'):
                    if path.is_file() and not any(x in path.parts for x in ['.git','__pycache__']):
                        tar.add(path,arcname=str(path.relative_to(ROOT)))
            for name in ['scene_train.json','scene_val.json']:
                p=ROOT/'data/annotations'/name;tar.add(p,arcname=str(p.relative_to(ROOT)))
            p=ROOT/'experiments/scene_groups/report.json';tar.add(p,arcname=str(p.relative_to(ROOT)))
        ssh = ' '.join(shlex.quote(v) for v in SSH[:-1])
        run('transfer_source',['rsync','-a','--info=progress2','-e',ssh,str(archive),HOST+':'+REMOTE+'/backups/scene_source.tar.gz'])
        run('transfer_environment',['rsync','-a','--info=progress2','-e',ssh,str(ROOT/'backups/AICOMP-env.tar.gz'),HOST+':'+REMOTE+'/backups/AICOMP-env.tar.gz'])
        run('transfer_images',['rsync','-a','--info=progress2','-e',ssh,str(ROOT/'data/train')+'/',HOST+':'+REMOTE+'/data/train/'])
        run('transfer_pretrained',['rsync','-a','--info=progress2','-e',ssh,str(ROOT/'checkpoints/dfine_x_obj2coco.pth'),HOST+':'+REMOTE+'/checkpoints/dfine_x_obj2coco.pth'])
        # All files live in a new task-owned directory. Never replace another user's environment.
        relocate = '''import json,os,subprocess,sys,tarfile,hashlib
from pathlib import Path
root=Path('/home2/fbohan/AIC');env=Path('/home2/fbohan/miniconda3/envs/AICOMP')
assert not (root/'experiments/scene_rgb_gpu6/controller.pid').exists(), 'Already launched'
with tarfile.open(root/'backups/scene_source.tar.gz') as tar:tar.extractall(root)
if not env.exists():
 env.mkdir(parents=True)
 with tarfile.open(root/'backups/AICOMP-env.tar.gz') as tar:tar.extractall(env)
 subprocess.run([str(env/'bin/python'),str(env/'bin/conda-unpack')],check=True)
else:
 assert (env/'bin/python').exists(), 'Do not overwrite a partial or unrelated environment'
for folder in ['scripts','configs']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.yml','.yaml','.sh'):
   s=p.read_text();s=s.replace('/home/fbohan/AIC','/home2/fbohan/AIC').replace('/home/fbohan/miniconda3','/home2/fbohan/miniconda3');p.write_text(s)
for n in ['scene_train','scene_val']:
 d=json.loads((root/'data/annotations'/f'{n}.json').read_text())
 assert all((root/'data/train'/i['file_name']).exists() for i in d['images'])
with (root/'checkpoints/dfine_x_obj2coco.pth').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
assert sha==PUBLIC_PRETRAIN_SHA
runtime={**os.environ,'LD_LIBRARY_PATH':str(env/'lib/python3.11/site-packages/nvidia/nvjitlink/lib')}
subprocess.run([str(env/'bin/python'),'-c','import torch,cv2,faster_coco_eval; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.device_count())'],env=runtime,check=True)
out=root/'experiments/scene_rgb_gpu6';out.mkdir(parents=True,exist_ok=True)
with (out/'controller.log').open('ab') as log:
 p=subprocess.Popen([str(env/'bin/python'),'-u',str(root/'scripts/run_scene_rgb.py')],cwd=root,env=runtime,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'controller.pid').write_text(str(p.pid));print('REMOTE_CONTROLLER',p.pid)
'''
        with (ROOT/'checkpoints/dfine_x_obj2coco.pth').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        relocate=relocate.replace('PUBLIC_PRETRAIN_SHA',repr(sha))
        command='python -c '+shlex.quote(relocate)
        run('relocate_and_launch',SSH+[command])
        status('remote_pipeline_started',remote_project=REMOTE,remote_host=HOST,
               note='Preflight and smoke must pass before formal training; not a completed training claim')
    except Exception as error:
        status('failed',error=repr(error))
        raise


if __name__ == '__main__':
    main()
