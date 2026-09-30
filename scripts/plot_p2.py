"""Refresh P2 and crop-control curves each minute without fake incomplete-epoch AP."""
import time,json,fcntl
from pathlib import Path
import plot_targetcrop as charts
ROOT=Path(__file__).resolve().parents[1];charts.OUT=ROOT/'monitoring/p2'
charts.RUNS={
 'p2_640':('P2检测特征 · 640','#2F5D9B','-',20),
 'continue800_control':('普通续训对照 · 800','#C47B32','--',15),
 'targetcrop800':('局部裁剪 · 800','#697349',':',15),
}
if __name__=='__main__':
 charts.OUT.mkdir(parents=True,exist_ok=True);lock=(charts.OUT/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  charts.generate()
  p=ROOT/'experiments/p2/status.json';s=json.loads(p.read_text()) if p.exists() else {}
  if s.get('stage')=='failed' or all((ROOT/'runs'/n/'COMPLETE').exists() for n in charts.RUNS):break
  time.sleep(60)
