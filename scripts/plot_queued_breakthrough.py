"""Minute-updated source-backed HTI/control curves; no interpolated results."""
import fcntl,json,os,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'monitoring/queued_breakthrough';OUT.mkdir(parents=True,exist_ok=True)
def draw():
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for name,label in [('ir_reliability800','IR reliability'),('ir_reliability_control800','IR gate off'),('roi_coverage800','ROI coverage'),('roi_coverage_control800','ROI top64')]:
        path=ROOT/'runs'/name/'metrics.jsonl';rows=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
        for axis,index,title in zip(list(axes.flat)[:4],[0,1,2,3],['mAP@50-95','AP50','AP75','small-object AP']):
            points=[]
            for row in rows:
                val=row.get('validation',row.get('val',{})).get('coco_eval_bbox',[])
                if len(val)>index:points.append((row.get('epoch',0),100*val[index]))
            if points:axis.plot(*zip(*points),marker='o',label=label)
            axis.set_title(title);axis.set_xlabel('epoch');axis.grid(alpha=.2)
        points=[]
        for row in rows:
            loss=row.get('train',{}).get('loss',row.get('train_loss'))
            if loss is not None:points.append((row.get('epoch',0),loss))
        if points:axes.flat[5].plot(*zip(*points),marker='o',label=label)
        ap90=[]
        for row in rows:
            val=row.get('validation',{}).get('ap_by_iou',{}).get('0.90')
            if val is not None:ap90.append((row.get('epoch',0),100*val))
        if ap90:axes.flat[4].plot(*zip(*ap90),marker='o',label=label)
    axes.flat[4].set_title('AP90');axes.flat[4].set_xlabel('epoch');axes.flat[4].grid(alpha=.2)
    axes.flat[5].set_title('train loss');axes.flat[5].set_xlabel('epoch');axes.flat[5].grid(alpha=.2)
    for ax in axes.flat:
        if ax.lines:ax.legend()
    fig.suptitle('Next mechanisms / matched controls | refreshed '+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S CST'))
    fig.tight_layout();tmp=OUT/f'overview.{os.getpid()}.png';fig.savefig(tmp,dpi=140);plt.close(fig);tmp.replace(OUT/'overview.png')
if __name__=='__main__':
    lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:draw()
        except Exception as e:print(repr(e),flush=True)
        time.sleep(60)
