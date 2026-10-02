"""Persistent GPU0 visual-patch experiment or GPU1 equal-budget RGB control."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/vf_tokens';GPU=1 if '--control-only' in sys.argv else 0
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':str(GPU),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
def status(stage,**fields):
    t=OUT/('control_status.tmp' if GPU==1 else 'status.tmp');t.write_text(json.dumps({'stage':stage,'gpu':GPU,'controller_pid':os.getpid(),'time':time.time(),**fields},indent=2));t.replace(OUT/('control_status.json' if GPU==1 else 'status.json'))
def free():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    return {int(a):int(b)for a,b in (x.split(',')for x in rows)}[GPU]
def main():
    OUT.mkdir(parents=True,exist_ok=True);child=None
    lock=(OUT/('control_controller.lock' if GPU==1 else 'controller.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/f'experiments/mechanism_trials/gpu{GPU}.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    def run(cmd,log,stage,**fields):
        nonlocal child
        with (OUT/log).open('a')as stream:child=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage,worker_pid=child.pid,**fields)
        if child.wait():raise RuntimeError(f'{stage} failed; inspect {log}')
    try:
        assert subprocess.check_output(['findmnt','-n','-T',str(ROOT/'checkpoints/gpu6_storage'),'-o','FSTYPE'],text=True).strip()=='fuse.sshfs'
        for name in (('vf_tokens_control800',) if GPU==1 else ('vf_tokens800',)):
            dest=ROOT/'checkpoints/gpu6_storage/vf_tokens/runs'/name;dest.mkdir(parents=True,exist_ok=True);local=ROOT/'runs'/name
            if local.is_symlink():assert local.resolve()==dest.resolve()
            elif local.exists():raise RuntimeError(f'Unexpected existing {local}')
            else:local.symlink_to(dest,target_is_directory=True)
        while free()<9216:status('waiting_for_memory',free_mib=free(),required_mib=9216);time.sleep(15)
        if GPU==1:
            while not (OUT/'preflight.json').exists():status('waiting_for_verified_preflight');time.sleep(15)
        if not (OUT/'preflight.json').exists():run([sys.executable,str(ROOT/'scripts/vf_tokens_preflight.py')],'preflight.log','preflight')
        assert json.loads((OUT/'preflight.json').read_text())['stage']=='passed'
        for name in (('vf_tokens_control800',) if GPU==1 else ('vf_tokens800',)):
            output=ROOT/'runs'/name
            if not (output/'COMPLETE').exists():
                if (output/'metrics.jsonl').exists():raise RuntimeError(f'Partial inference-only run needs new destination: {name}')
                while free()<9216:status('waiting_for_memory',run=name,free_mib=free());time.sleep(15)
                cmd=[sys.executable,str(ROOT/'scripts/vf_tokens_train.py'),'--config',str(ROOT/'configs'/f'{name}.yml'),'--init-checkpoint',str(ROOT/'runs/ft_aug800/weights_epoch_020.pth')]
                run(cmd+['--smoke'],f'{name}_smoke.log','smoke',run=name)
                run(cmd+['--evaluate-init'],f'{name}.log','training',run=name,epochs=8,batch_size=1,effective_batch=8,memory_limit_gib=8.5,queued=['visual_feature_ablations'] if name=='vf_tokens800'else[])
                assert (output/'COMPLETE').exists()
        if GPU==0 and not (OUT/'ablations_COMPLETE').exists():
            run([sys.executable,str(ROOT/'scripts/vf_tokens_ablations.py'),'--checkpoint',str(ROOT/'runs/vf_tokens800/best.pth')],'ablations.log','ablations')
            (OUT/'ablations_COMPLETE').write_text('done\n')
        status('complete',runs=['vf_tokens_control800'] if GPU==1 else ['vf_tokens800'])
    except BaseException:status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll()is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        gpu.close();lock.close()
if __name__=='__main__':main()
