"""Run independent DEIMv2 candidates and matched context-view trials overnight."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def main(job):
    out=ROOT/'experiments'/('contextmix' if job=='context' else 'deimv2')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/(job+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**details):
        tmp=out/(job+'_status.tmp');tmp.write_text(json.dumps({'stage':stage,'time':time.strftime('%F %T'),**details},indent=2));tmp.replace(out/(job+'_status.json'))
    def wait_memory(gpus,required):
        status('waiting_for_memory',gpus=gpus)
        while True:
            q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
            mem={int(p[0]):int(p[1]) for line in q.splitlines() if (p:=line.split(','))}
            if all(mem[g]>=required for g in gpus):return
            time.sleep(10)
    def run(name,gpus,command,required=9900):
        if (ROOT/'runs'/name/'COMPLETE').exists():return
        if (ROOT/'runs'/name/'metrics.jsonl').exists():raise RuntimeError('Partial run requires explicit resume: '+name)
        wait_memory(gpus,required)
        env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))}
        status('smoke',run=name,gpus=gpus)
        with (out/(name+'_smoke.log')).open('a') as log:subprocess.run(command+['--smoke'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/(name+'.log')).open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('training',run=name,gpus=gpus,pid=p.pid)
        if p.wait():raise RuntimeError('Training failed: '+name)
        if not (ROOT/'runs'/name/'COMPLETE').exists():raise RuntimeError('Missing COMPLETE: '+name)
    try:
        if job=='context':
            for i,name in enumerate(['contextmix800','contextfull800']):
                cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3',f'--master_port={29971+i}',str(ROOT/'scripts/train_baseline.py'),'--config',str(ROOT/f'configs/{name}.yml'),'--init-checkpoint',str(ROOT/'runs/ft_aug800/weights_epoch_020.pth'),'--evaluate-init']
                run(name,[0,2,5],cmd)
            from after_targetcrop import choose,records
            ann=json.loads((ROOT/'data/annotations/val400.json').read_text())
            counts={c['name']:sum(a['category_id']==c['id'] for a in ann['annotations']) for c in ann['categories']}
            initial=json.loads((ROOT/'runs/contextmix800/initial_metrics.json').read_text())['coco_eval_bbox']
            decision=choose(records('contextmix800'),records('contextfull800'),initial,counts)
            (out/'decision.json').write_text(json.dumps(decision,indent=2))
            status('comparison_ready',decision=decision,note='No automatic full-data promotion based only on val400.')
        else:
            name='deimv2_'+job;gpu=7 if job=='l' else 4
            run(name,[gpu],[sys.executable,str(ROOT/'scripts/train_deimv2.py'),'--config',str(ROOT/f'configs/{name}.yml')],8900)
            status('training_complete',run=name,note='24-epoch independent-val400 screening; phase2 score unknown.')
    except Exception:
        status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',choices=['context','l','x'],required=True)
    main(parser.parse_args().job)
