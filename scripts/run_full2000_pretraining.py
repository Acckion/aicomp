"""Prepare two full-data candidates while shadow validation runs independently."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['pool','reset'],required=True);p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--large-probe',action='store_true');a=p.parse_args()
    assert not a.large_probe or a.method=='pool'
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    out=ROOT/'experiments/full2000_pretraining'/a.method;out.mkdir(parents=True,exist_ok=True)
    own=(out/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/f'experiments/mechanism_trials/gpu{a.gpu_index}.lock').open('a');child=None
    def status(stage,**extra):
        t=out/'status.tmp';t.write_text(json.dumps({'stage':stage,'time':time.time(),'controller_pid':os.getpid(),'method':a.method,**extra},indent=2));t.replace(out/'status.json')
    def memory_ready(stage):
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=(9216 if a.large_probe and name.endswith('_b2') else 7168):return
            status('waiting_memory',next_stage=stage,free_mib=free);time.sleep(15)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu_index),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    def run(stage,command):
        nonlocal child
        memory_ready(stage)
        with (out/f'{stage}.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage,worker_pid=child.pid,run_name=name)
        return child.wait()
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    try:
        batch=1
        probe=ROOT/'experiments'/('scene_batch_probe_large' if a.large_probe else 'scene_batch_probe')
        while True:
            if (probe/'comparison.json').exists():
                comparison=json.loads((probe/'comparison.json').read_text())
                for b in [1,2]:
                    record=json.loads((probe/f'batch{b}.json').read_text())
                    assert record['stage']=='passed' and record['actual_finite_updates']==3
                if comparison['batch2_speed_ratio']>=1.1 and comparison['batch2_peak_mib']<=(8192 if a.large_probe else 5800):batch=2
                break
            if (probe/'status.json').exists():
                record=json.loads((probe/'status.json').read_text())
                if record['stage'] in ['failed_probe','failed']:
                    assert json.loads((probe/'batch1.json').read_text())['stage']=='passed'
                    break
            status('waiting_batch_probe');time.sleep(15)
        while True:
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:status('waiting_existing_task');time.sleep(15)
        for selected in ([2,1] if batch==2 else [1]):
            name=f'full2000_obj365_{a.method}800'+('_b2' if selected==2 else '')
            output=ROOT/'runs'/name
            assert not (output/'metrics.jsonl').exists() and not (output/'COMPLETE').exists()
            output.mkdir(parents=True,exist_ok=True)
            policy={'training_images':2000,'epochs':100,'effective_batch':8,'selected_micro_batch':selected,
                    'source':'public Objects365-only X; no competition parent weights',
                    'validation':False,'shadow_run':'scene_obj365_'+a.method+'800',
                    'candidate_snapshot_epochs':[40,60,80,100],
                    'selection':'Compare corresponding schedule stages using independent shadow training and official feedback. Full-data AP is unavailable; no train-set best claim.',
                    'note':'Parallel full-data hypothesis trial, not confirmation of shadow or phase2 improvement.'}
            (output/'selection_policy.json').write_text(json.dumps(policy,indent=2))
            config=ROOT/'configs'/(name+'.yml')
            code=run('preflight_b'+str(selected),[str(ROOT/'scripts/scene_semantic_preflight.py'),'--name',name])
            if code:
                if selected==2:
                    assert not (output/'metrics.jsonl').exists();continue
                raise RuntimeError('Full-data batch1 preflight failed')
            result=json.loads((ROOT/'experiments/scene_semantic_init'/name/'preflight.json').read_text())
            assert result['training_images']==2000 and result['heldout_images']==0 and result['validation_disabled']
            command=[str(ROOT/'scripts/train_ir_content.py'),'--config',str(config),'--init-checkpoint',str(ROOT/'checkpoints/dfine_x_obj365.pth')]
            code=run('smoke_b'+str(selected),command+['--smoke'])
            if code:
                if selected==2:
                    assert not (output/'metrics.jsonl').exists();continue
                raise RuntimeError('Full-data batch1 smoke failed')
            if run('training',command):raise RuntimeError('Full-data training failed; no automatic restart')
            assert (output/'COMPLETE').exists();status('complete',run_name=name);return
        raise RuntimeError('No viable full-data microbatch')
    except Exception as e:status('failed',error=repr(e));raise
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()


if __name__=='__main__':main()
