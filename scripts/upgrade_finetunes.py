"""Switch single-GPU jobs at an epoch checkpoint to six shared GPUs."""
import os,sys,time,signal,subprocess,json,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/finetune'
ENV={**os.environ,'OMP_NUM_THREADS':'3','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
SOURCES={640:102884,800:102885};DEVICES={640:[0,2,5],800:[3,4,6]}

def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def update(state):
    tmp=OUT/'upgrade_status.tmp';tmp.write_text(json.dumps({'time':time.strftime('%F %T'),'jobs':state},indent=2));tmp.replace(OUT/'upgrade_status.json')

def main():
    lock=(OUT/'upgrade.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={str(s):{'stage':'waiting_for_epoch_checkpoint','source_pid':pid,'gpus':DEVICES[s],'batch_per_gpu':3,'global_batch':9,'memory_limit_gib':8.5} for s,pid in SOURCES.items()}
    pending=set(SOURCES);active={};update(state)
    while pending or active:
        for size in list(pending):
            checkpoint=ROOT/f'runs/ft_aug{size}/last.pth'
            if not checkpoint.exists():
                if not live(SOURCES[size]):
                    state[str(size)]['stage']='source_failed_without_checkpoint';pending.remove(size);update(state)
                continue
            pid=SOURCES[size]
            if live(pid):
                command=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
                if 'train_baseline.py' not in command:raise RuntimeError('PID identity changed')
                os.kill(pid,signal.SIGTERM)
                for _ in range(20):
                    if not live(pid):break
                    time.sleep(1)
                if live(pid):raise RuntimeError(f'Source {pid} did not exit cleanly')
            state[str(size)]['stage']='multi_gpu_smoke'
            env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,DEVICES[size]))}
            command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3',f'--master_port={29801 if size==640 else 29802}',str(ROOT/'scripts/train_baseline.py'),'--config',str(ROOT/f'configs/ft_aug{size}_shared3.yml'),'--resume',str(checkpoint),'--rebatch-resume']
            log=(OUT/f'aug{size}_three_gpu_smoke.log').open('w')
            proc=subprocess.Popen(command+['--smoke'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[size]=(proc,log,'smoke',command,env);pending.remove(size)
            state[str(size)]['pid']=proc.pid;update(state)
        for size,(proc,log,phase,command,env) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del active[size]
            if phase=='smoke' and code==0:
                log=(OUT/f'aug{size}_three_gpu.log').open('a')
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[size]=(proc,log,'training',command,env)
                state[str(size)].update(stage='training',pid=proc.pid,exit_code=None)
            else:state[str(size)].update(stage='completed' if code==0 else phase+'_failed',exit_code=code)
            update(state)
        if pending or active:time.sleep(5)

if __name__=='__main__':
    try:main()
    except Exception:
        (OUT/'upgrade_failure.txt').write_text(traceback.format_exc());raise
