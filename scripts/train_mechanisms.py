"""Register experiment components, then reuse the complete D-FINE training runner."""
import argparse
import fcntl
import importlib
from pathlib import Path
import signal
import sys

import torch
import train_baseline as baseline
from mechanism_runtime import check_bn_behavior, install_training_controls, install_checkpoint_controls
from src.core.yaml_utils import load_config


def main():
    # A launcher that temporarily masks signals must not leave its worker
    # immune to graceful termination after exec.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    parser.add_argument('--check-bn', action='store_true')
    args, _ = parser.parse_known_args()
    if args.check_bn:
        check_bn_behavior()
        return
    if not args.config:
        raise ValueError('--config is required')
    config = load_config(args.config)
    for name in config.get('mechanism_imports', []):
        importlib.import_module(name)
    output = Path(config['output_dir'])
    lock_dir = baseline.ROOT / 'experiments/mechanism_trials'
    lock_dir.mkdir(parents=True, exist_ok=True)
    # The detached worker owns this lock. A controller can exit while the
    # worker remains alive, so a controller-only lock cannot prevent duplicates.
    worker_lock = (lock_dir / f'{output.name}.train.lock').open('a')
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f'Another live training worker owns this run: {output}') from error
    if (output / 'COMPLETE').exists():
        raise RuntimeError(f'Refusing to overwrite completed run: {output}')
    if (output / 'metrics.jsonl').exists() and '--resume' not in sys.argv:
        raise RuntimeError(f'Partial run needs explicit --resume: {output}')
    install_training_controls(baseline)
    install_checkpoint_controls(baseline, config)
    try:
        baseline.main()
    finally:
        if torch.cuda.is_initialized():
            print(f'CUDA_PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated() / 1024**2:.2f}', flush=True)
        worker_lock.close()


if __name__ == '__main__':
    main()
