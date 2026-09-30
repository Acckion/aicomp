"""Wait without occupying GPUs, then run the approved baseline ablations."""
import os,json,time,subprocess,sys,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/simple';OUT.mkdir(parents=True,exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'2','LD_LIBRARY_PATH':'/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'}

def status(stage,**kwargs):
    (OUT/'status.json').write_text(json.dumps({'stage':stage,'time':time.strftime('%F %T'),**kwargs},indent=2))

def command(size,tile,name,checkpoint):
    return [sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(checkpoint),'--size',str(size),'--tile',str(tile),'--output',str(OUT/name)]

def main():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not all((ROOT/'runs'/n/'COMPLETE').exists() for n in ('rgb1600','rgb2000')):
        # No repeated epoch/log inspection; wait at most one check per 20 minutes.
        status('waiting_for_baselines')
        time.sleep(1200)
    # Do not compete with another GPU user's compute jobs.
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        status('waiting_for_idle_gpus');time.sleep(1200)
    record=json.loads((ROOT/'runs/rgb1600/status.json').read_text())
    checkpoint=ROOT/f'runs/rgb1600/weights_epoch_{record["best_epoch"]:03d}.pth'
    status('smoke_test',checkpoint=str(checkpoint))
    smoke=command(640,0,'smoke',checkpoint)+['--limit','2']
    with (OUT/'smoke.log').open('a') as log:
        subprocess.run(smoke,env={**ENV,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    variants=[(640,0),(800,0),(960,0),(1120,0),(640,.6),(800,.6),(960,.6),(640,.5)]
    jobs=[]
    for gpu,(size,tile) in enumerate(variants):
        name=f'size{size}_tile{tile:g}'
        if (OUT/name/'COMPLETE').exists():continue
        log=(OUT/(name+'.log')).open('a')
        p=subprocess.Popen(command(size,tile,name,checkpoint),env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,p,log))
    status('evaluating',checkpoint=str(checkpoint),jobs={n:p.pid for n,p,_ in jobs})
    failures={}
    for name,p,log in jobs:
        code=p.wait();log.close()
        if code: failures[name]=code
    rows=[]
    for size,tile in variants:
        path=OUT/f'size{size}_tile{tile:g}'/'results.json'
        if path.exists():
            result=json.loads(path.read_text())
            rows.extend(dict(size=size,tile=tile,**r) for r in result['results'])
    rows.sort(key=lambda r:r['map'],reverse=True)
    (OUT/'ranking.json').write_text(json.dumps(rows,indent=2))
    lines=['# 固定最佳 RGB1600 权重的推理对照','',f'权重：{checkpoint}','独立 val400；不是榜单分数。每图最终最多 100 框。','', '|尺寸|切片比例（0=整图）|后处理|阈值|mAP@50–95|','|---|---|---|---|---|']
    for r in rows:lines.append(f'|{r["size"]}|{r["tile"]}|{r["method"]}|{r["threshold"]}|{r["map"]:.4f}|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    status('completed' if not failures else 'partial_failure',failures=failures,best=rows[0] if rows else None)

if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',traceback=traceback.format_exc());raise
