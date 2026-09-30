"""Run target-crop trial, then its matched continuation control on the released crop GPUs."""
import os,sys,time,json,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/targetcrop';OUT.mkdir(parents=True,exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'3','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}
WEIGHTS=ROOT/'runs/ft_aug800/weights_epoch_020.pth'

def status(stage,**details):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'time':time.strftime('%F %T'),'stage':stage,**details},indent=2));tmp.replace(OUT/'status.json')

def run(name,gpus,port):
    if (ROOT/'runs'/name/'COMPLETE').exists():return
    if (ROOT/'runs'/name/'metrics.jsonl').exists():raise RuntimeError(f'Partial run {name}; explicit resume required')
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3',f'--master_port={port}',str(ROOT/'scripts/train_baseline.py'),'--config',str(ROOT/f'configs/{name}.yml'),'--init-checkpoint',str(WEIGHTS)]
    env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))}
    status('smoke',run=name,gpus=gpus)
    with (OUT/(name+'_smoke.log')).open('a') as log:
        subprocess.run(cmd+['--smoke'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (OUT/(name+'.log')).open('a') as log:
        p=subprocess.Popen(cmd+['--evaluate-init'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('training',run=name,gpus=gpus,pid=p.pid)
        code=p.wait()
        if code:raise RuntimeError(f'{name} failed with exit code {code}')


def main():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run('targetcrop800',[0,2,5],29911)
    run('continue800_control',[0,2,5],29912)
    reports={}
    for name in ('targetcrop800','continue800_control'):
        rows=[json.loads(s) for s in (ROOT/'runs'/name/'metrics.jsonl').read_text().splitlines()]
        best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0]);last=rows[-1]
        reports[name]={'best_epoch':best['epoch'],'best_map':best['validation']['coco_eval_bbox'][0]*100,'best_stats':best['validation']['coco_eval_bbox'],'per_class':best['validation']['per_class_ap'],'last5_maps':[r['validation']['coco_eval_bbox'][0]*100 for r in rows[-5:]],'final_map':last['validation']['coco_eval_bbox'][0]*100}
    (OUT/'comparison.json').write_text(json.dumps(reports,indent=2));status('comparison_ready',report=str(OUT/'comparison.json'))

if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',traceback=traceback.format_exc());raise
