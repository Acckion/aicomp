"""Plot whole-epoch AP, class AP and losses each minute for the night trials."""
import fcntl
import json
from pathlib import Path
import time
import plot_targetcrop as charts

ROOT=Path(__file__).resolve().parents[1]
charts.OUT=ROOT/'monitoring/night_experiments'
charts.RUNS={
    'deimv2_l':('DEIMv2-L · 640','#2F5D9B','-',24),
    'deimv2_x':('DEIMv2-X · 640','#705B89','-',24),
    'contextmix800':('上下文局部混合 · 800','#697349','-',12),
    'contextfull800':('整图续训对照 · 800','#C47B32','--',12),
}

if __name__=='__main__':
    charts.OUT.mkdir(parents=True,exist_ok=True)
    lock=(charts.OUT/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        charts.generate()
        states=[ROOT/'experiments/deimv2/l_status.json',ROOT/'experiments/deimv2/x_status.json',ROOT/'experiments/contextmix/context_status.json']
        terminal=[p.exists() and json.loads(p.read_text()).get('stage') in ('failed','training_complete','comparison_ready') for p in states]
        if all(terminal):break
        time.sleep(60)
