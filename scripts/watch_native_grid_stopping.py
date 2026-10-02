"""Stop verified owned weak pairs after four complete matched epochs."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native_grid'


def rows(name):
    p=ROOT/'runs'/name/'metrics.jsonl'
    result={}
    if p.exists():
        for line in p.read_text().splitlines():
            try:r=json.loads(line);result[r['epoch']]=r
            except json.JSONDecodeError:pass
    return result


def owned(pid,needle):
    p=Path(f'/proc/{pid}/cmdline')
    try:
        args=[a.decode() for a in p.read_bytes().split(b'\0') if a]
        if needle.endswith('.py'):return any(Path(a).name==needle for a in args)
        return needle in args or any(Path(a).name==needle+'.yml' for a in args)
    except FileNotFoundError:return False


def main():
    lock=(OUT/'early_stop_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending={'native_grid1','native_grid2'}
    while pending:
        for name in list(pending):
            control=name+'_control';a,b=rows(name),rows(control)
            if not all(e in a and e in b for e in [3,4]):continue
            if not all((ROOT/'runs'/run/'weights_epoch_004.pth').exists() for run in [name,control]):continue
            comparison=[]
            for e in [3,4]:
                av,bv=a[e]['validation']['coco_eval_bbox'],b[e]['validation']['coco_eval_bbox']
                comparison.append({'epoch':e,'map_delta':100*(av[0]-bv[0]),'small_ap_delta':100*(av[3]-bv[3])})
            decision={'paired_epochs':comparison,'threshold':'both epochs3/4 map worse by>.5 and no small AP gain>.5',
                      'stop':all(r['map_delta']<-.5 and r['small_ap_delta']<=.5 for r in comparison),'time':time.time(),
                      'note':'Budget stop, not completed8 epochs; no test labels or submitted score selects weights'}
            if decision['stop']:
                for run in [name,control]:
                    state=json.loads((OUT/run/'status.json').read_text())
                    cp,wp=state.get('controller_pid'),state.get('worker_pid')
                    did_stop=False
                    if cp and owned(cp,'run_native_grid.py') and owned(cp,run):
                        os.kill(cp,signal.SIGTERM);did_stop=True
                    elif wp and owned(wp,'train_native_grid.py') and owned(wp,run):
                        assert os.getpgid(wp)==wp
                        os.killpg(wp,signal.SIGTERM);did_stop=True
                    decision.setdefault('stopped_owned_runs' if did_stop else 'no_matching_live_owned_process',[]).append(run)
            (OUT/name/'early_stop_decision.json').write_text(json.dumps(decision,indent=2))
            print(name,json.dumps(decision),flush=True);pending.remove(name)
        if pending:time.sleep(60)


if __name__=='__main__':main()
