"""Prepare one predeclared full-data candidate only after held-out evidence gates."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/phase2_obj365_pool_e40'
RUN=ROOT/'runs/full2000_obj365_pool800_b2'
AUDIT=ROOT/'experiments/taxonomy_submission_audit_e10/report.json'
SHADOW=ROOT/'checkpoints/gpu8_storage/runs'


def records(path):
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return {r['epoch']:r for r in [json.loads(x) for x in lines if x]}


def evidence():
    pool=records(SHADOW/'scene_obj365_pool800/metrics.jsonl')
    reset=records(SHADOW/'scene_obj365_reset800/metrics.jsonl')
    common=sorted(set(pool)&set(reset))
    epochs=[e for e in common if e>=4][-3:]
    if len(epochs)<3:return None
    delta=[];without_sparse=[]
    for e in epochs:
        a=pool[e]['validation'];b=reset[e]['validation']
        delta.append((a['coco_eval_bbox'][0]-b['coco_eval_bbox'][0])*100)
        values=[(a['per_class_ap'][c]-b['per_class_ap'][c])*100
                for c in a['per_class_ap'] if c!='tricycle']
        without_sparse.append(sum(values)/len(values))
    return {'common_epochs':epochs,'map_deltas':delta,
            'mean_map_delta':sum(delta)/3,
            'mean_delta_excluding_sparse_tricycle':sum(without_sparse)/3,
            'supported':sum(delta)/3>=.5 and sum(without_sparse)/3>0}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    owner=(OUT/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        p=OUT/'status.tmp';p.write_text(json.dumps({'stage':stage,'time':time.time(),
            'controller_pid':os.getpid(),**extra},indent=2));p.replace(OUT/'status.json')
    phase=json.loads((ROOT/'data/phase2/images.json').read_text())
    assert len(phase['images'])==1000 and 'annotations' not in phase
    names={Path(i['file_name']).stem+'.txt' for i in phase['images']}
    assert len(names)==1000 and all((ROOT/'data/phase2'/i['file_name']).is_file() for i in phase['images'])
    full=json.loads((ROOT/'data/annotations/train2000.json').read_text())
    assert len(full['images'])==2000
    checkpoint=RUN/'weights_epoch_040.pth'
    while True:
        try:
            if not AUDIT.exists():status('waiting_validation_route_audit');time.sleep(60);continue
            audit=json.loads(AUDIT.read_text())
            if not audit['route_consistent']:
                status('rejected_route_mismatch',audit=audit);return
            if not (RUN/'metrics.jsonl').exists() or 40 not in records(RUN/'metrics.jsonl'):
                status('waiting_full_epoch40');time.sleep(60);continue
            assert checkpoint.is_file()
            decision=evidence()
            if decision is None:
                status('waiting_matched_post_warmup_epochs');time.sleep(60);continue
            if not decision['supported']:
                status('rejected_shadow_evidence',evidence=decision);return
            break
        except OSError as error:
            status('observation_retry',error=repr(error));time.sleep(60)
    (OUT/'selection_evidence.json').write_text(json.dumps({'validation_audit':audit,
        'shadow_comparison':decision,'fixed_epoch':40,'full_training_images':2000,
        'limitations':'Preparation gate only; grouped validation is not phase2 ground truth or proof of57. No automatic submission.'},indent=2))
    gpu=(ROOT/'experiments/mechanism_trials/gpu4.lock').open('a')
    while True:
        try:
            fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
            free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free>=3072:break
            fcntl.flock(gpu,fcntl.LOCK_UN);status('waiting_memory',free_mib=free)
        except BlockingIOError:status('waiting_project_gpu')
        time.sleep(30)
    command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_variants.py'),
        '--checkpoint',str(checkpoint),'--config',str(ROOT/'configs/full2000_obj365_pool800_b2.yml'),
        '--annotations',str(ROOT/'data/phase2/images.json'),'--image-root',str(ROOT/'data/phase2'),
        '--size','800','--amp','--native-top100','--require-ema','--method','none',
        '--predict-only','--gpu-memory-limit-gib','2','--output',str(OUT/'prediction')]
    with (OUT/'prediction.log').open('ab') as log:
        worker=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'4',
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT)
        status('generating_candidate',worker_pid=worker.pid)
        code=worker.wait()
    if code:status('failed',returncode=code);raise RuntimeError('Inference failed; no blind retry')
    package=OUT/'prediction/submission.zip'
    with zipfile.ZipFile(package) as z:
        assert len(z.namelist())==1000 and set(z.namelist())==names and z.testzip() is None
    manifest=json.loads((OUT/'prediction/submission_manifest.json').read_text())
    assert manifest['images']==1000 and manifest['size']==800
    assert manifest['candidate_selection']=='native_top100_no_refill_v1'
    assert manifest['precision']=='cuda_float16_v1' and manifest['leaderboard_score'] is None
    status('candidate_ready',package=str(package),leaderboard_score=None,
           note='Prepared only; review held-out trend and current official feedback before manual submission.')


if __name__=='__main__':main()
