"""Publish completed GPU7 prerequisites; retire only the suspended own queue."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/gpu8_migration'
SSH = ['ssh', '-i', '/home/fbohan/.ssh/aicomp_servers_ed25519', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
       'fbohan@222.20.97.217']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'publisher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    done = set()
    while len(done) < 2:
        for direction in ['ir', 'roi']:
            if direction in done:
                continue
            names = ['ir_content800', 'ir_content_control800'] if direction == 'ir' else ['roi_coverage800']
            if not all((ROOT / 'runs' / name / 'COMPLETE').exists() for name in names):
                continue
            if direction == 'ir':
                state = json.loads((ROOT / 'experiments/ir_content/status.json').read_text())
                if state.get('stage') != 'complete':
                    continue
            if direction == 'roi':
                record = json.loads((ROOT / 'experiments/next_roi/migration_pause.json').read_text())
                pid = record['controller_pid']
                proc = Path('/proc') / str(pid)
                if proc.exists():
                    assert proc.stat().st_uid == os.getuid()
                    assert 'run_queued_breakthrough.py' in (proc / 'cmdline').read_text()
                    # TERM remains pending during STOP; resume delivers it before
                    # the old controller can launch its next queued worker.
                    os.kill(pid, signal.SIGTERM); os.kill(pid, signal.SIGCONT)
            payload = {'stage': 'complete', 'direction': direction, 'completed_runs': names, 'source_server': 'GPU7', 'time': time.time()}
            encoded = json.dumps(payload)
            script = 'from pathlib import Path\np=Path("/home/fbohan/AIC/migration/upstream/' + direction + '.ready.json")\np.with_suffix(".tmp").write_text(' + repr(encoded) + ')\np.with_suffix(".tmp").replace(p)\n'
            try:
                result = subprocess.run(SSH + ['python3', '-'], input=script, text=True, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                print('Prerequisite SSH timeout, retry:', direction, flush=True)
                continue
            if result.returncode == 0:
                (OUT / (direction + '_published.json')).write_text(encoded); done.add(direction)
            else:
                print('Prerequisite publication failed, retry:', direction, result.stderr, flush=True)
        time.sleep(15)


if __name__ == '__main__':
    main()
