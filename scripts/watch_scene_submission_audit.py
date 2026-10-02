"""Audit a fixed shadow checkpoint against its full held-out submission conversion."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'scene_obj365_pool800'
EPOCH = 10
RUN = ROOT / 'checkpoints/gpu8_storage/runs' / NAME
OUT = ROOT / 'experiments/taxonomy_submission_audit_e10'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    owner = (OUT / 'controller.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        p = OUT / 'status.tmp'
        p.write_text(json.dumps({'stage': stage, 'time': time.time(),
                                 'controller_pid': os.getpid(), **extra}, indent=2))
        p.replace(OUT / 'status.json')
    checkpoint = RUN / f'weights_epoch_{EPOCH:03d}.pth'
    train = json.loads((ROOT / 'data/annotations/scene_train.json').read_text())
    val = json.loads((ROOT / 'data/annotations/scene_val.json').read_text())
    assert len(train['images']) == 1610 and len(val['images']) == 390
    assert {i['id'] for i in train['images']}.isdisjoint(i['id'] for i in val['images'])
    assert len({Path(i['file_name']).stem for i in val['images']}) == 390
    row = None
    while row is None:
        try:
            rows = [json.loads(x) for x in (RUN / 'metrics.jsonl').read_text().splitlines() if x]
            row = next((r for r in rows if r['epoch'] == EPOCH), None)
            if row is not None:
                assert checkpoint.is_file()
                break
            status('waiting_checkpoint',epoch=EPOCH,completed_epochs=len(rows))
        except OSError as error:
            status('observation_retry', error=repr(error))
        time.sleep(60)
    gpu = (ROOT / 'experiments/mechanism_trials/gpu4.lock').open('a')
    while True:
        try:
            fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free = int(subprocess.check_output(['nvidia-smi','-i','4',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free >= 3072:
                break
            fcntl.flock(gpu, fcntl.LOCK_UN)
            status('waiting_memory', free_mib=free)
        except BlockingIOError:
            status('waiting_project_gpu')
        time.sleep(30)
    command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_variants.py'),
               '--checkpoint',str(checkpoint),'--config',str(ROOT/'configs'/f'{NAME}.yml'),
               '--annotations',str(ROOT/'data/annotations/scene_val.json'),
               '--image-root',str(ROOT/'data/train'),'--size','800','--amp',
               '--native-top100','--require-ema','--single-method','--method','none',
               '--gpu-memory-limit-gib','2','--output',str(OUT/'evaluation')]
    with (OUT/'evaluation.log').open('ab') as log:
        worker = subprocess.Popen(command,cwd=ROOT,env={**os.environ,
            'CUDA_VISIBLE_DEVICES':'4','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
            stdout=log,stderr=subprocess.STDOUT)
        status('evaluating',worker_pid=worker.pid,epoch=EPOCH)
        code=worker.wait()
    if code:
        status('failed',returncode=code)
        raise RuntimeError('Audit failed; no automatic retry or training restart')
    result=json.loads((OUT/'evaluation/results.json').read_text())
    assert result['images']==390 and not result['limited_subset']
    assert len(list((OUT/'evaluation/validation_txt').glob('*.txt')))==390
    native=result['unclipped_reference']['map']
    clipped=result['results'][0]['map']
    txt=result['results'][0]['txt_roundtrip_map']
    training=row['validation']['coco_eval_bbox'][0]*100
    report={'epoch':EPOCH,'images':390,'training_validation_map':training,
            'standalone_native_map':native,'clipped_map':clipped,'txt_map':txt,
            'standalone_minus_training':native-training,'clipping_delta':clipped-native,
            'txt_roundtrip_delta':txt-clipped,
            'route_consistent':abs(native-training)<.05 and abs(clipped-native)<.05 and abs(txt-clipped)<.01,
            'limitations':'Fixed shadow validation only. Clipping changes are reported separately. No phase2 predictions, ZIP, or leaderboard improvement claim.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    status('complete' if report['route_consistent'] else 'route_mismatch',report=report)


if __name__=='__main__':
    main()
