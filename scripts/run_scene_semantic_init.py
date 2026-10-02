"""Memory-bounded grouped RGB training on a dedicated remote project copy."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]



def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['scene_semantic_init800','scene_rgb_fastcontrol800','scene_obj365_reset800','scene_obj365_pool800','scene_obj365_pool800_twowheel','scene_obj365_pool800_airqueries','scene_obj365_pool800_subtypedn'],required=True);p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--resume');p.add_argument('--minimum-free-mib',type=int,default=7168);args=p.parse_args()
    OUT=ROOT/'experiments/scene_semantic_init'/args.name;OUT.mkdir(parents=True,exist_ok=True)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    lock = (ROOT/f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(args.gpu_index),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
           'LD_LIBRARY_PATH':str(Path(sys.prefix)/'lib/python3.11/site-packages/nvidia/nvjitlink/lib')}
    def status(stage, **extra):
        tmp = OUT / 'status.tmp'
        tmp.write_text(json.dumps({'stage':stage,'time':time.time(),'controller_pid':os.getpid(),**extra},indent=2))
        tmp.replace(OUT / 'status.json')
    def run(stage, command):
        nonlocal child
        while True:
            free = int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free >= args.minimum_free_mib:
                break
            status('waiting_memory', next_stage=stage, free_mib=free)
            time.sleep(15)
        with (OUT / f'{stage}.log').open('ab') as log:
            child = subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage, worker_pid=child.pid)
        if child.wait():
            raise RuntimeError(f'{stage} failed, inspect its log')
    signal.signal(signal.SIGTERM, lambda n,f:sys.exit(128+n))
    signal.signal(signal.SIGINT, lambda n,f:sys.exit(128+n))
    try:
        assert not (ROOT / 'runs' / args.name / 'COMPLETE').exists()
        command = [str(ROOT/'scripts/train_ir_content.py'),'--config',str(ROOT/'configs'/(args.name+'.yml'))]
        if args.resume:
            assert Path(args.resume).is_file()
            assert json.loads((OUT/'preflight.json').read_text())['actual_updates'] == 3
            command += ['--resume',args.resume]
        else:
            if args.name.endswith('_airqueries'):
                run('coverage', [str(ROOT/'scripts/audit_air_query_coverage.py'), '--config',str(ROOT/'configs'/(args.name+'.yml')), '--output',str(OUT/'coverage')])
                if not json.loads((OUT/'coverage/report.json').read_text())['launch_supported']:
                    status('rejected_geometry_gate', run_name=args.name)
                    return
            if args.name.endswith('_subtypedn'):
                run('subtype_support', [str(ROOT/'scripts/audit_subtype_dn.py'), '--config',str(ROOT/'configs'/(args.name+'.yml')), '--output',str(OUT/'subtype_support')])
                if not json.loads((OUT/'subtype_support/report.json').read_text())['launch_supported']:
                    status('rejected_subtype_support_gate', run_name=args.name)
                    return
            run('preflight', [str(ROOT/'scripts/scene_semantic_preflight.py'),'--name',args.name])
            assert json.loads((OUT/'preflight.json').read_text())['actual_updates'] == 3
            if args.name.startswith('scene_obj365_'):
                command += ['--init-checkpoint',str(ROOT/'checkpoints/dfine_x_obj365.pth')]
            run('smoke', command + ['--smoke'])
        run('training', command)
        assert (ROOT/'runs'/args.name/'COMPLETE').exists()
        status('complete')
    except Exception as error:
        status('failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            child.wait()


if __name__ == '__main__':
    main()
