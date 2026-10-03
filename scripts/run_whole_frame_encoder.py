"""Launch a preflighted encoder adaptation beside one identified owned job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=['scene_whole_encoder_real', 'scene_whole_encoder_sham'], required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--peer-pid', type=int, required=True)
    parser.add_argument('--peer-config', required=True)
    parser.add_argument('--output-target', type=Path, required=True)
    args = parser.parse_args()
    from src.core.yaml_utils import load_config
    config_path = ROOT / 'configs' / (args.name+'.yml')
    config = load_config(str(config_path), {})
    cap = float(config['gpu_memory_limit_gib'])
    out = ROOT / 'experiments/whole_frame_head' / args.name
    out.mkdir(parents=True, exist_ok=True)
    owner = (out/'controller.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    buddy = (ROOT/f'experiments/mechanism_trials/gpu{args.gpu_index}.whole_encoder_buddy.lock').open('a')
    fcntl.flock(buddy, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = json.loads((out/'preflight_max_gt.json').read_text())
    assert preflight['stage'] == 'passed' and preflight['name'] == args.name
    assert preflight['train_encoder'] and preflight['encoder_parameters_changed']
    assert preflight['frozen_backbone_and_running_stats_exact'] and preflight['strict_reload_exact']
    assert preflight['maximum_annotation_count_test'] and preflight['sham_reconstruction_identity_exact']
    assert preflight['memory_cap_gib'] <= cap <= 10
    child = None
    def status(stage, **extra):
        p = out/'status.tmp'
        p.write_text(json.dumps({'stage':stage, 'controller_pid':os.getpid(), 'name':args.name,
            'gpu':args.gpu_index, 'peer_pid':args.peer_pid, 'memory_cap_gib':cap, 'time':time.time(), **extra}, indent=2))
        p.replace(out/'status.json')
    signal.signal(signal.SIGTERM, lambda n,f: sys.exit(128+n))
    signal.signal(signal.SIGINT, lambda n,f: sys.exit(128+n))
    try:
        while True:
            command = Path(f'/proc/{args.peer_pid}/cmdline').read_text()
            assert args.peer_config in command, 'Peer must remain the identified owned training job'
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            rows = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index),
                '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
            peers = [int(r.split(',')[1].strip()) for r in rows if r.split(',')[0].strip() == str(args.peer_pid)]
            assert len(peers) == 1
            ceiling = peers[0]+int(cap*1024)+1024
            if free >= int(cap*1024)+512 and ceiling <= 10240:
                break
            status('waiting_shared_budget', free_mib=free, peer_mib=peers[0], combined_ceiling_mib=ceiling)
            time.sleep(15)
        train_path = Path(config['train_dataloader']['dataset']['ann_file'])
        val_path = Path(config['val_dataloader']['dataset']['ann_file'])
        train, val = json.loads(train_path.read_text()), json.loads(val_path.read_text())
        assert len(train['images']) == 1610 and len(val['images']) == 390
        assert {i['id'] for i in train['images']}.isdisjoint(i['id'] for i in val['images'])
        run, target = Path(config['output_dir']), args.output_target
        target.mkdir(parents=True, exist_ok=True)
        if run.exists() or run.is_symlink():
            assert run.resolve() == target.resolve()
        else:
            run.symlink_to(target, target_is_directory=True)
        assert not (run/'metrics.jsonl').exists() and not (run/'COMPLETE').exists(), 'No duplicate training or blind restart'
        checkpoint = ROOT/'runs/scene_obj365_pool800/weights_epoch_020.pth'
        source_sha = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
        assert source_sha == 'bef5fa932b425d2f5991e45450c98fe1097c978f9272fabdc5266ad42f42668f'
        manifest = {'parent_checkpoint':str(checkpoint), 'parent_sha256':source_sha,
            'train_annotation_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
            'val_annotation_sha256':hashlib.sha256(val_path.read_bytes()).hexdigest(),
            'geometry':[1088,1920], 'mode':config['WholeFrameHeadDFINE']['mode'],
            'trainable':'encoder and decoder; backbone and BN running statistics frozen',
            'epochs':8, 'warmup_epochs':1, 'batch':1, 'accumulation':8,
            'decoder_lr':config['optimizer']['lr'], 'encoder_lr':0.00001,
            'preflight':preflight, 'combined_ceiling_mib_at_launch':ceiling,
            'comparison':'Same parent, fold, seed, geometry and schedule. Real uses original pixels; sham reconstructs from800-square RGB. Compare with frozen-encoder arms to test feature adaptation.',
            'limitations':'One seed; separate devices and shared compute. No phase2 fitting or online gain claim.'}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        with (out/'training.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/train_ir_content.py'),
                '--config', str(config_path), '--init-checkpoint', str(checkpoint), '--evaluate-init'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES':str(args.gpu_index),
                    'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'})
            status('training', worker_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError(f'Training failed with{code}; no blind retry')
        assert (run/'COMPLETE').exists()
        status('complete')
    except BaseException:
        status('failed', traceback=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    sys.path[:0] = [str(ROOT/'D-FINE'), str(ROOT/'scripts')]
    main()
