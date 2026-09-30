"""After fine-tunes finish, verify gains, transfer to full data and package predictions."""
import os,sys,time,json,subprocess,fcntl,traceback,statistics
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/next_stage';OUT.mkdir(parents=True,exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def write(stage,**details):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'stage':stage,'time':time.strftime('%F %T'),**details},indent=2));tmp.replace(OUT/'status.json')


def records(name):
    return [json.loads(s) for s in (ROOT/'runs'/name/'metrics.jsonl').read_text().splitlines()]


def choose_training():
    base=max(records('rgb1600'),key=lambda r:r['validation']['coco_eval_bbox'][0])
    reference=base['validation']['coco_eval_bbox'][0]*100
    ann=json.loads((ROOT/'data/annotations/val400.json').read_text())
    counts={c['name']:sum(a['category_id']==c['id'] for a in ann['annotations']) for c in ann['categories']}
    candidates=[]
    for size in (640,800):
        rows=records(f'ft_aug{size}');best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0])
        recent=rows[-5:];initial=json.loads((ROOT/f'runs/ft_aug{size}/initial_metrics.json').read_text())['coco_eval_bbox']
        overall=best['validation']['coco_eval_bbox'][0]*100
        recent_map=statistics.median(r['validation']['coco_eval_bbox'][0]*100 for r in recent)
        recent_small=statistics.median(r['validation']['coco_eval_bbox'][3]*100 for r in recent)
        class_delta={n:(best['validation']['per_class_ap'][n]-base['validation']['per_class_ap'][n])*100 for n in counts}
        severe=[n for n,v in class_delta.items() if counts[n]>=30 and v < -5]
        # Operational screening, not a statistical significance claim.
        passes=(overall>=reference+.1 and recent_map>=reference-.2 and recent_small>=initial[3]*100-.5 and len(severe)<=1)
        candidates.append({'size':size,'best_epoch':best['epoch'],'best_map':overall,'gain_vs_original640':overall-reference,'median_last5_map':recent_map,'median_last5_small_ap':recent_small,'initial_small_ap':initial[3]*100,'class_delta_vs_original640':class_delta,'common_class_drops_over5':severe,'passes_screen':passes})
    passed=[c for c in candidates if c['passes_screen']]
    selected=max(passed,key=lambda c:(c['median_last5_small_ap'],c['median_last5_map'])) if passed else None
    decision={'reference_map':reference,'candidates':candidates,'selected':selected,'note':'val400 screening only; phase2 is harder. Thresholds are operational, not proof of generalization or statistical significance.'}
    (OUT/'training_decision.json').write_text(json.dumps(decision,indent=2));return selected


def spawn(cmd,gpus,logname):
    log=(OUT/logname).open('a')
    p=subprocess.Popen(cmd,cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return p,log


def check(p,log):
    code=p.wait();log.close()
    if code:raise RuntimeError(f'Child process {p.pid} exited {code}; inspect experiment logs')


def free_memory(gpus,minimum=9900):
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    memory={int(parts[0]):int(parts[1]) for line in q.splitlines() if (parts:=line.split(','))}
    return all(memory[g]>=minimum for g in gpus)


def main():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write('waiting_for_finetunes')
    while not all((ROOT/f'runs/ft_aug{s}/COMPLETE').exists() for s in (640,800)):
        status_path=ROOT/'experiments/finetune/upgrade_status.json'
        if status_path.exists():
            jobs=json.loads(status_path.read_text()).get('jobs',{})
            if any(j.get('exit_code') not in (None,0) for j in jobs.values()):
                raise RuntimeError('A source fine-tune failed; stopping dependent experiments')
        time.sleep(5)
    # Let distributed workers exit before reusing their share of memory.
    while not free_memory([0,2,3,4,5,6]):time.sleep(30)
    selected=choose_training()
    if selected is None:
        write('no_stable_gain_found',decision=str(OUT/'training_decision.json'))
        return
    size=selected['size'];epoch=selected['best_epoch']
    checkpoint=ROOT/f'runs/ft_aug{size}/weights_epoch_{epoch:03d}.pth'
    full_name=f'ft2000_aug{size}'
    cfg=yaml.safe_load((ROOT/f'configs/ft_aug{size}_shared3.yml').read_text())
    cfg['validate']=False;cfg['output_dir']=str(ROOT/'runs'/full_name)
    cfg['train_dataloader']['dataset']['ann_file']=str(ROOT/'data/annotations/train2000.json')
    cfg_path=OUT/'full2000.yml';cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Matching baseline phase is a proxy, not a measured optimum for full-data training.
    full_start=ROOT/'runs/rgb2000/weights_epoch_089.pth'
    train=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3','--master_port=29901',str(ROOT/'scripts/train_baseline.py'),'--config',str(cfg_path),'--init-checkpoint',str(full_start)]
    write('full_data_smoke',selected=selected)
    p,log=spawn(train+['--smoke'],[3,4,6],'full2000_smoke.log');check(p,log)
    full_p,full_log=spawn(train,[3,4,6],'full2000_train.log')
    write('training_full_data_and_testing_inference',full_data_pid=full_p.pid,selected=selected)
    variants=[(640,0),(800,0),(960,0),(1120,0),(640,.6),(800,.6)]
    # Three CPU/GPU workers each evaluate two configurations, while full-data trains.
    pending=list(variants);active={};results=[];failures=[]
    while pending or active:
        for gpu,(p,log,name) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close();del active[gpu]
            if code:failures.append({'variant':name,'exit_code':code})
            else:
                result=json.loads((OUT/name/'results.json').read_text())
                for row in result['results']:results.append({'size':result['size'],'tile':result['tile_fraction'],**row})
        for gpu in (0,2,5):
            if not pending:break
            if gpu in active or not free_memory([gpu]):continue
            res,tile=pending.pop(0);name=f'infer{res}_tile{tile:g}'
            cmd=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(checkpoint),'--size',str(res),'--tile',str(tile),'--output',str(OUT/name)]
            p,log=spawn(cmd,[gpu],name+'.log');active[gpu]=(p,log,name)
        if full_p.poll() not in (None,0):raise RuntimeError('Full-data training failed')
        if pending or active:time.sleep(5)
    if not results:raise RuntimeError('All inference variants failed')
    best_map=max(r['map'] for r in results)
    # Within 0.1 point of best overall, prefer stronger small-object AP.
    shortlist=[r for r in results if r['map']>=best_map-.1]
    best=max(shortlist,key=lambda r:(r['stats'][3],-r['size'],r['tile']==0))
    (OUT/'inference_decision.json').write_text(json.dumps({'selected':best,'all_results':sorted(results,key=lambda r:r['map'],reverse=True),'failures':failures,'note':'Local validation choice; verify phase2 score with an actual submission.'},indent=2))
    write('waiting_for_full_data_training',full_data_pid=full_p.pid,inference=best)
    check(full_p,full_log)
    if not (ROOT/'runs'/full_name/'COMPLETE').exists():raise RuntimeError('Missing full-data COMPLETE marker')
    # Export baseline, matching fine-tune phase, and final phase as separate candidates.
    candidates=[('baseline2000',ROOT/'runs/rgb2000/weights_epoch_089.pth',640,0,'none',0),
                ('finetune_matching_phase',ROOT/f'runs/{full_name}/weights_epoch_{epoch:03d}.pth',best['size'],best['tile'],best['method'],best['threshold']),
                ('finetune_final',ROOT/f'runs/{full_name}/weights_epoch_020.pth',best['size'],best['tile'],best['method'],best['threshold'])]
    write('generating_phase2_packages')
    active=[]
    for gpu,(name,weights,res,tile,method,threshold) in zip((0,2,5),candidates):
        while not free_memory([gpu]):time.sleep(30)
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(weights),'--size',str(res),'--tile',str(tile),'--method',method,'--threshold',str(threshold),'--predict-only','--annotations',str(ROOT/'data/phase2/images.json'),'--image-root',str(ROOT/'data/phase2'),'--output',str(OUT/name)]
        p,log=spawn(cmd,[gpu],name+'.log');active.append((p,log))
    for p,log in active:check(p,log)
    write('packages_ready_for_submission',packages=[str(OUT/name/'submission.zip') for name,*_ in candidates],leaderboard_score=None,selected=selected,inference=best)

if __name__=='__main__':
    try:main()
    except Exception:
        trace=traceback.format_exc();write('failed',traceback=trace);raise
