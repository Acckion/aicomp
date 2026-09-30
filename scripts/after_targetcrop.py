"""After matched crop/control runs finish, screen gains and launch next ablations."""
import os, sys, json, time, fcntl, statistics, subprocess, traceback, argparse
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/after_targetcrop'
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def write(stage,**details):
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'stage':stage,'time':time.strftime('%F %T'),**details},indent=2));tmp.replace(OUT/'status.json')


def records(name):
    raw=(ROOT/'runs'/name/'metrics.jsonl').read_bytes()
    return [json.loads(s) for s in raw.splitlines(keepends=True) if s.endswith(b'\n')]


def choose(crop,control,initial,counts):
    def med(rows,index):return statistics.median(r['validation']['coco_eval_bbox'][index]*100 for r in rows[-5:])
    crop_best=max(crop,key=lambda r:r['validation']['coco_eval_bbox'][0])
    control_best=max(control,key=lambda r:r['validation']['coco_eval_bbox'][0])
    gain=med(crop,0)-med(control,0);small_gain=med(crop,3)-med(control,3)
    deltas={name:statistics.median(r['validation']['per_class_ap'][name]*100 for r in crop[-5:])-statistics.median(r['validation']['per_class_ap'][name]*100 for r in control[-5:]) for name in counts}
    drops=[n for n,d in deltas.items() if counts[n]>=100 and d < -2]
    # Operational thresholds, not a significance test or phase2 guarantee.
    passes=gain>=.1 and small_gain>=.25 and med(crop,0)>=initial[0]*100 and not drops
    return {'transfer_crop':passes,'crop_best_epoch':crop_best['epoch'],'crop_best_map':crop_best['validation']['coco_eval_bbox'][0]*100,'control_best_map':control_best['validation']['coco_eval_bbox'][0]*100,'crop_median_last5_map':med(crop,0),'control_median_last5_map':med(control,0),'median_last5_map_gain':gain,'median_last5_small_ap_gain':small_gain,'class_delta_last5':deltas,'common_class_drops_over2':drops,'note':'Relative val400 screening only; not significance, not phase2 performance. No score-based selection on full-data training.'}


def free(gpus):
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    mem={int(p[0]):int(p[1]) for line in q.splitlines() if (p:=line.split(','))}
    return all(mem[g]>=9900 for g in gpus)


def launch(config,weights,gpus,port,name,eval_init=False):
    if (ROOT/'runs'/name/'metrics.jsonl').exists():raise RuntimeError(f'{name} already exists; explicit resume required')
    while not free(gpus):time.sleep(5)
    env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))}
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3',f'--master_port={port}',str(ROOT/'scripts/train_baseline.py'),'--config',str(config),'--init-checkpoint',str(weights)]
    with (OUT/(name+'_smoke.log')).open('a') as f:
        subprocess.run(cmd+['--smoke'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (OUT/(name+'.log')).open('a') as f:
        p=subprocess.Popen(cmd+(['--evaluate-init'] if eval_init else []),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return p


def main(skip_control=False):
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if skip_control:
        if not (ROOT/'runs/targetcrop800/COMPLETE').exists():
            raise RuntimeError('Crop experiment must finish before proceeding')
        decision={'transfer_crop':False,'control_skipped_by_user':True,'note':'User requested immediate structural optimization. No paired continuation attribution or crop promotion decision is claimed.'}
    else:
        write('waiting_for_crop_control_and_packages')
        while True:
            p=ROOT/'experiments/targetcrop/status.json';s=json.loads(p.read_text()) if p.exists() else {}
            q=ROOT/'experiments/next_stage/status.json';t=json.loads(q.read_text()) if q.exists() else {}
            if s.get('stage')=='failed' or t.get('stage')=='failed':raise RuntimeError('Dependency failed; inspect upstream status before continuing')
            if all((ROOT/'runs'/n/'COMPLETE').exists() for n in ('targetcrop800','continue800_control')) and t.get('stage')=='packages_ready_for_submission':break
            time.sleep(5)
        ann=json.loads((ROOT/'data/annotations/val400.json').read_text())
        counts={c['name']:sum(a['category_id']==c['id'] for a in ann['annotations']) for c in ann['categories']}
        initial=json.loads((ROOT/'runs/targetcrop800/initial_metrics.json').read_text())['coco_eval_bbox']
        decision=choose(records('targetcrop800'),records('continue800_control'),initial,counts)
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    jobs=[]
    with (OUT/'plot.log').open('a') as log:
        plot=subprocess.Popen([sys.executable,str(ROOT/'scripts/plot_targetcrop.py'),'--next','--watch'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'plot.pid').write_text(str(plot.pid))
    # Always keep the structural ablation independent of target cropping.
    write('detail_smoke',decision=decision)
    p=launch(ROOT/'configs/detail800.yml',ROOT/'runs/ft_aug800/weights_epoch_020.pth',[0,2,5],29921,'detail800',True)
    jobs.append(('detail800',p))
    write('detail_training',pid=p.pid,decision=decision)
    if decision['transfer_crop']:
        cfg=yaml.safe_load((ROOT/'configs/targetcrop800.yml').read_text())
        cfg['validate']=False;cfg['output_dir']=str(ROOT/'runs/targetcrop2000')
        cfg['train_dataloader']['dataset']['ann_file']=str(ROOT/'data/annotations/train2000.json')
        config=OUT/'targetcrop2000.yml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        p=launch(config,ROOT/'runs/ft2000_aug800/weights_epoch_020.pth',[3,4,6],29922,'targetcrop2000')
        jobs.append(('targetcrop2000',p))
        write('detail_and_crop_full_training',jobs={n:p.pid for n,p in jobs},decision=decision)
    for name,p in jobs:
        code=p.wait()
        if code:raise RuntimeError(f'{name} exited {code}')
        if not (ROOT/'runs'/name/'COMPLETE').exists():raise RuntimeError(f'{name} missing COMPLETE')
    if decision['transfer_crop']:
        epoch=decision['crop_best_epoch']
        while not free([3]):time.sleep(5)
        env={**ENV,'CUDA_VISIBLE_DEVICES':'3'}
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(ROOT/f'runs/targetcrop2000/weights_epoch_{epoch:03d}.pth'),'--size','800','--method','none','--predict-only','--annotations',str(ROOT/'data/phase2/images.json'),'--image-root',str(ROOT/'data/phase2'),'--output',str(OUT/'crop_phase2')]
        write('packaging_crop_phase2',epoch=epoch,decision=decision)
        with (OUT/'crop_phase2.log').open('a') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write('next_training_complete',decision=decision,jobs=[n for n,p in jobs],crop_package=str(OUT/'crop_phase2/submission.zip') if decision['transfer_crop'] else None,leaderboard_score=None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skip-control',action='store_true');args=parser.parse_args()
    try:main(args.skip_control)
    except Exception:
        write('failed',traceback=traceback.format_exc());raise
