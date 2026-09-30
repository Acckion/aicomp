"""Start each fine-tune as soon as one GPU is idle; never terminate other jobs."""
import os,sys,time,json,subprocess,fcntl,traceback
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/finetune';OUT.mkdir(parents=True,exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'4','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
CHECKPOINT=ROOT/'runs/rgb1600/weights_epoch_089.pth'
POLL_SECONDS=5
SHARED = os.environ.get("AICOMP_SHARED_GPU") == "1"

def idle_gpus():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
    occupied={line.strip() for line in apps.splitlines()}
    if SHARED:
        candidates=[(int(p[0].strip()),int(p[2])) for line in query.splitlines() if (p:=line.split(',')) and int(p[2])>=10240]
        return [index for index,free in sorted(candidates,key=lambda item:item[1],reverse=True)]
    return [int(p[0].strip()) for line in query.splitlines() if (p:=line.split(',')) and p[1].strip() not in occupied and int(p[2])>22000]

def single_gpu_config(size):
    cfg=yaml.safe_load((ROOT/f'configs/ft_aug{size}.yml').read_text())
    cfg['__include__']=[str(ROOT/'configs/rgb1600.yml')]
    cfg['train_dataloader']['total_batch_size']=4
    cfg['val_dataloader']['total_batch_size']=4
    if SHARED:
        cfg['train_dataloader']['total_batch_size']=1
        cfg['val_dataloader']['total_batch_size']=1
        cfg['gradient_accumulation_steps']=4
        cfg['gpu_memory_limit_gib']=7.5
    # Same batch and LR in both resolution trials; linear scaling from batch 16.
    cfg['optimizer']['lr']=.0000125
    cfg['optimizer']['params'][0]['lr']=.000000125
    path=OUT/f'aug{size}_single_gpu.yml'
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return path

def save(stage,jobs,pending,available=None,error=None):
    record={'stage':stage,'time':time.strftime('%F %T'),'checkpoint':str(CHECKPOINT),
            'poll_seconds':POLL_SECONDS,'training_review_seconds':1200,'shared_gpu':SHARED,
            'pending_sizes':pending,'idle_gpus':available,'jobs':jobs}
    if error:record['last_gpu_query_error']=error
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(OUT/'status.json')

def main():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[s for s in (640,800) if not (ROOT/f'runs/ft_aug{s}/COMPLETE').exists()]
    active={};state={};last_signature=None
    configs={s:single_gpu_config(s) for s in pending}
    for s in pending:
        if (ROOT/f'runs/ft_aug{s}/metrics.jsonl').exists():
            raise RuntimeError(f'Partial existing run ft_aug{s}; explicit resume is required')
    while pending or active:
        for size,(p,log,gpu,phase) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close();del active[size]
            if phase=='smoke' and code==0:
                env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)}
                log=(OUT/f'aug{size}.log').open('a')
                cmd=[sys.executable,str(ROOT/'scripts/train_baseline.py'),'--config',str(configs[size]),'--init-checkpoint',str(CHECKPOINT),'--evaluate-init']
                p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[size]=(p,log,gpu,'training')
                state[str(size)].update(stage='training',pid=p.pid,exit_code=None)
            else:
                state[str(size)].update(stage=('completed' if code==0 else phase+'_failed'),exit_code=code)
        error=None
        try:available=idle_gpus()
        except (subprocess.SubprocessError,OSError) as exc:
            available=[];error=str(exc)
        reserved={gpu for _,_,gpu,_ in active.values()}
        available=[g for g in available if g not in reserved]
        while pending and available:
            size=pending.pop(0);gpu=available.pop(0)
            env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)}
            log=(OUT/f'aug{size}_smoke.log').open('a')
            cmd=[sys.executable,str(ROOT/'scripts/train_baseline.py'),'--config',str(configs[size]),'--init-checkpoint',str(CHECKPOINT),'--smoke']
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            active[size]=(p,log,gpu,'smoke')
            state[str(size)]={'stage':'smoke','gpus':[gpu],'pid':p.pid,'exit_code':None,'batch_size':1 if SHARED else 4,'gradient_accumulation_steps':4 if SHARED else 1,'gpu_memory_limit_gib':7.5 if SHARED else None,'learning_rate':.0000125}
        signature=json.dumps([state,pending,available,error],sort_keys=True)
        if signature!=last_signature:
            stage='running' if active else ('waiting_for_idle_gpus' if pending else 'finished')
            save(stage,state,pending,available,error);last_signature=signature
        if pending or active:time.sleep(POLL_SECONDS)

if __name__=='__main__':
    try:main()
    except Exception:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise
