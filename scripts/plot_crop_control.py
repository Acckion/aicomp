"""Plot the completed crop run and newly requested matched continuation control."""
import json,time,fcntl
from pathlib import Path
import plot_targetcrop as charts
ROOT=Path(__file__).resolve().parents[1]
charts.OUT=ROOT/'monitoring/crop_control'
charts.RUNS={
 'targetcrop800':('小目标裁剪 · 1600','#2F5D9B','-',15),
 'continue800_control':('普通续训对照 · 1600','#C47B32','--',15),
}
if __name__=='__main__':
 charts.OUT.mkdir(parents=True,exist_ok=True);lock=(charts.OUT/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  charts.generate()
  p=ROOT/'experiments/crop_control/status.json';s=json.loads(p.read_text()) if p.exists() else {}
  if s.get('stage') in ('comparison_ready','failed'):break
  time.sleep(60)
