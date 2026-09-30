"""Resume packaging after an independently running full-data training finishes."""
import os,sys,time,json,subprocess,fcntl,traceback,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/next_stage'
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}


def write(stage,**details):
    p=OUT/'status.tmp';p.write_text(json.dumps({'stage':stage,'time':time.strftime('%F %T'),**details},indent=2));p.replace(OUT/'status.json')


def free(gpu):
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    m={int(p[0]):int(p[1]) for line in q.splitlines() if (p:=line.split(','))}
    return m[gpu]>=9900


def main(pid):
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=json.loads((OUT/'training_decision.json').read_text())['selected']
    inference=json.loads((OUT/'inference_decision.json').read_text())['selected']
    name=f"ft2000_aug{selected['size']}"
    write('waiting_for_full_data_training',full_data_pid=pid,inference=inference,packaging_gpus=[3,4,6])
    while not (ROOT/'runs'/name/'COMPLETE').exists():
        if not Path(f'/proc/{pid}').exists():raise RuntimeError('Full-data process exited without COMPLETE')
        time.sleep(5)
    candidates=[('baseline2000',ROOT/'runs/rgb2000/weights_epoch_089.pth',640,0,'none',0),
                ('finetune_matching_phase',ROOT/f"runs/{name}/weights_epoch_{selected['best_epoch']:03d}.pth",inference['size'],inference['tile'],inference['method'],inference['threshold']),
                ('finetune_final',ROOT/f'runs/{name}/weights_epoch_020.pth',inference['size'],inference['tile'],inference['method'],inference['threshold'])]
    # Do not spend inference time or submissions on identical checkpoint/settings.
    unique=[];seen=set()
    for item in candidates:
        key=(str(item[1].resolve()),*item[2:])
        if key not in seen:unique.append(item);seen.add(key)
    write('generating_phase2_packages',packaging_gpus=[3,4,6],candidates=[c[0] for c in unique])
    jobs=[]
    for gpu,(label,weights,size,tile,method,threshold) in zip([3,4,6],unique):
        while not free(gpu):time.sleep(5)
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(weights),'--size',str(size),'--tile',str(tile),'--method',method,'--threshold',str(threshold),'--predict-only','--annotations',str(ROOT/'data/phase2/images.json'),'--image-root',str(ROOT/'data/phase2'),'--output',str(OUT/label)]
        with (OUT/(label+'.log')).open('a') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append((label,p))
    for label,p in jobs:
        if p.wait():raise RuntimeError(f'Packaging failed: {label}')
        if not (OUT/label/'submission.zip').exists():raise RuntimeError(f'Missing package: {label}')
    write('packages_ready_for_submission',packages=[str(OUT/c[0]/'submission.zip') for c in unique],leaderboard_score=None,selected=selected,inference=inference)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-pid',type=int,required=True);args=parser.parse_args()
    try:main(args.training_pid)
    except Exception:
        write('failed',traceback=traceback.format_exc());raise
