"""Native trainer with a train-only prototype retrieval residual."""
import argparse
import fcntl
import signal
import torch
import train_baseline as baseline
import extra_iou_metrics
import instance_memory
from mechanism_runtime import install_training_controls,install_checkpoint_controls
from src.core.yaml_utils import load_config

def main():
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',required=True)
    args,_=parser.parse_known_args();config=load_config(args.config)
    output=baseline.Path(config['output_dir'])
    lock=(baseline.ROOT/'experiments/instance_memory'/f'{output.name}.train.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'COMPLETE').exists():raise RuntimeError('Completed experiment exists')
    if (output/'metrics.jsonl').exists() and '--resume' not in baseline.sys.argv:raise RuntimeError('Resume required')
    instance_memory.install(enabled=config.get('instance_memory_enabled',True))
    instance_memory.install_initialization(baseline)
    install_training_controls(baseline);install_checkpoint_controls(baseline,config)
    try:baseline.main()
    finally:
        if torch.cuda.is_initialized():print(f'CUDA_PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated()/1024**2:.2f}',flush=True)
        lock.close()
if __name__=='__main__':main()
