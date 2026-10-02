"""Persistent GPU1 original-pixel ROI and equal-budget disabled-ROI queue."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/native_roi';GPU=1
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':str(GPU),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
def status(stage,**fields):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'stage':stage,'time':time.time(),'controller_pid':os.getpid(),'gpu':GPU,**fields},indent=2));tmp.replace(OUT/'status.json')
def free():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    return {int(a):int(b)for a,b in (x.split(',')for x in lines)}[GPU]
def main():
    OUT.mkdir(parents=True,exist_ok=True);child=None
    controller=(OUT/'controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/'experiments/mechanism_trials/gpu1.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    def run(cmd,log,stage,**fields):
        nonlocal child
        with (OUT/log).open('a') as stream:child=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage,pid=child.pid,**fields)
        if child.wait():raise RuntimeError(f'{stage} failed: {log}')
    try:
        assert subprocess.check_output(['findmnt','-n','-T',str(ROOT/'checkpoints/gpu6_storage'),'-o','FSTYPE'],text=True).strip()=='fuse.sshfs'
        for name in ['native_roi800','native_roi_control800']:
            destination=ROOT/'checkpoints/gpu6_storage/native_roi/runs'/name;destination.mkdir(parents=True,exist_ok=True);local=ROOT/'runs'/name
            if local.is_symlink():assert local.resolve()==destination.resolve()
            elif local.exists():raise RuntimeError(f'Unexpected existing directory {local}')
            else:local.symlink_to(destination,target_is_directory=True)
        while free()<9216:status('waiting_for_memory',free_mib=free());time.sleep(15)
        preflight=OUT/'preflight.json'
        if not preflight.exists():run([sys.executable,str(ROOT/'scripts/native_roi_preflight.py')],'preflight.log','preflight')
        assert json.loads(preflight.read_text())['stage']=='passed'
        for name in ['native_roi800','native_roi_control800']:
            output=ROOT/'runs'/name
            if (output/'COMPLETE').exists():continue
            if (output/'metrics.jsonl').exists():raise RuntimeError(f'Partial inference-only run requires new destination {name}')
            cmd=[sys.executable,str(ROOT/'scripts/train_native_roi.py'),'--config',str(ROOT/'configs'/f'{name}.yml'),'--init-checkpoint',str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')]
            run(cmd+['--smoke'],f'{name}_smoke.log','smoke',run=name)
            run(cmd+['--evaluate-init'],f'{name}.log','training',run=name,epochs=8,effective_batch=8,per_gpu_batch=2,memory_limit_gib=8.5,queued_after_this=['native_roi_control800'] if name=='native_roi800' else [])
            assert (output/'COMPLETE').exists()
        status('complete',runs=['native_roi800','native_roi_control800'])
    except BaseException:status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        gpu.close();controller.close()
if __name__=='__main__':main()
