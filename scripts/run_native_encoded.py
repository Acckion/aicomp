"""Paired detail continuation controller with a checked shared-GPU budget."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['scene_native_encoded_delta800','scene_native_encoded_control800'],required=True)
 p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--peer-pid',type=int,required=True);p.add_argument('--peer-config',required=True)
 p.add_argument('--output-target',required=True);a=p.parse_args()
 out=ROOT/'experiments/native_encoded'/a.name;out.mkdir(parents=True,exist_ok=True)
 owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 buddy=(ROOT/f'experiments/mechanism_trials/gpu{a.gpu_index}.native_encoded_buddy.lock').open('a');fcntl.flock(buddy,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from src.core.yaml_utils import load_config
 config=load_config(str(ROOT/'configs'/f'{a.name}.yml'), {});cap=float(config['gpu_memory_limit_gib'])
 checkpoint=ROOT/'runs/scene_obj365_pool800/weights_epoch_020.pth';assert checkpoint.is_file()
 train=json.loads(Path(config['train_dataloader']['dataset']['ann_file']).read_text());val=json.loads(Path(config['val_dataloader']['dataset']['ann_file']).read_text())
 assert len(train['images'])==1610 and len(val['images'])==390
 assert {i['id'] for i in train['images']}.isdisjoint(i['id'] for i in val['images'])
 target=Path(a.output_target);target.mkdir(parents=True,exist_ok=True);run=Path(config['output_dir'])
 if run.is_symlink():assert run.resolve()==target.resolve()
 elif run.exists():assert run.resolve()==target.resolve()
 else:run.symlink_to(target,target_is_directory=True)
 assert not (run/'metrics.jsonl').exists() and not (run/'COMPLETE').exists(),'Existing run needs explicit recovery'
 child=None
 def status(stage,**kw):
  f=out/'status.tmp';f.write_text(json.dumps({'stage':stage,'name':a.name,'gpu':a.gpu_index,'controller_pid':os.getpid(),'peer_pid':a.peer_pid,'memory_cap_gib':cap,'time':time.time(),**kw},indent=2));f.replace(out/'status.json')
 def launch(stage,extra):
  nonlocal child
  while True:
   try:
    assert a.peer_config in Path(f'/proc/{a.peer_pid}/cmdline').read_text(),'Shared-GPU peer no longer matches expected owned task'
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    rows=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    peer=[int(r.split(',')[1].strip()) for r in rows if r.split(',')[0].strip()==str(a.peer_pid)];assert len(peer)==1
    ceiling=peer[0]+int(cap*1024)+1024
    if free>=int(cap*1024)+1536 and ceiling<=10240:break
    status('waiting_shared_budget',next_stage=stage,free_mib=free,peer_mib=peer[0],combined_ceiling_mib=ceiling)
   except subprocess.CalledProcessError as error:status('observation_retry',error=str(error))
   time.sleep(15)
  cmd=[sys.executable,'-u',str(ROOT/'scripts/train_ir_content.py'),'--config',str(ROOT/'configs'/f'{a.name}.yml'),'--init-checkpoint',str(checkpoint),*extra]
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu_index),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
  with (out/f'{stage}.log').open('ab') as f:child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  status(stage,worker_pid=child.pid);code=child.wait()
  if code:raise RuntimeError(f'{stage} failed ({code}); no blind restart')
 signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
 try:
  manifest={'parent_checkpoint':str(checkpoint.resolve()),'sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),'parent_training_epochs':20,'train_annotation_sha256':hashlib.sha256(Path(config['train_dataloader']['dataset']['ann_file']).read_bytes()).hexdigest(),'validation_annotation_sha256':hashlib.sha256(Path(config['val_dataloader']['dataset']['ann_file']).read_bytes()).hexdigest(),'train_images':1610,'validation_images':390,'epochs':8,'batch':1,'accumulation':8,'validation_is_not_phase2':True,'comparison':'Same packed RGB augmentation, schedule and initialization; residual disabled in control. Allocation caps differ only for device headroom.','rules':'Official training images only; one detector, no box voting or test fitting.'}
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
  launch('smoke',['--smoke']);launch('training',['--evaluate-init'])
  assert (run/'COMPLETE').exists();status('complete')
 except BaseException:
  status('failed',traceback=traceback.format_exc());raise
 finally:
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   except ProcessLookupError:pass

if __name__=='__main__':
 sys.path[:0]=[str(ROOT/'D-FINE'),str(ROOT/'scripts')];main()
