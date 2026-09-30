"""Automatically diagnose DEIMv2 candidates after their completed training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/deimv2_followup'
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def status(stage,**details):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'stage':stage,'time':time.strftime('%F %T'),**details},indent=2));tmp.replace(OUT/'status.json')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reports={}
    for size,gpu in [('l',7),('x',4)]:
        name='deimv2_'+size
        status('waiting_for_training',run=name)
        while not (ROOT/'runs'/name/'COMPLETE').exists():
            state=ROOT/f'experiments/deimv2/{size}_status.json'
            if state.exists() and json.loads(state.read_text()).get('stage')=='failed':
                reports[name]={'failed_training':True};break
            time.sleep(10)
        if reports.get(name,{}).get('failed_training'):continue
        rows=[json.loads(s) for s in (ROOT/'runs'/name/'metrics.jsonl').read_text().splitlines()]
        assert len(rows)==24,'Expected complete 24-epoch model screening'
        best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0])
        reports[name]={'best_epoch':best['epoch'],'diagnoses':{}}
        for resolution,epoch,label in [(640,best['epoch'],'best'),(800,best['epoch'],'best'),(640,24,'last')]:
            # Deduplicate when the final epoch is also the best.
            key=f'{epoch}_{resolution}'
            if key in reports[name]['diagnoses']:continue
            while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())<8900:time.sleep(10)
            folder=OUT/f'{name}_{key}';env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)}
            status('evaluating',run=name,epoch=epoch,resolution=resolution)
            command=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--backend','deimv2','--config',str(ROOT/f'configs/{name}.yml'),'--checkpoint',str(ROOT/f'runs/{name}/weights_epoch_{epoch:03d}.pth'),'--size',str(resolution),'--output',str(folder),'--single-method','--method','none','--gpu-memory-limit-gib','8']
            with (OUT/'evaluation.log').open('a') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (OUT/'diagnosis.log').open('a') as log:subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_validation.py'),'--predictions',str(folder/'raw_predictions.json'),'--output',str(folder/'diagnostics'),'--source',f'{name} epoch{epoch} {resolution} same-val400 plain'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((folder/'diagnostics/report.json').read_text())
            reports[name]['diagnoses'][key]={'checkpoint_role':label,'cohorts':report['cohorts']}
        (OUT/'comparison.json').write_text(json.dumps({'candidates':reports,'note':'Val400 model screening and overlapping GT cohorts; no phase2 score, no full-data auto-promotion.'},indent=2))
    status('comparison_ready',report=str(OUT/'comparison.json'))


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);status('failed',traceback=traceback.format_exc());raise
