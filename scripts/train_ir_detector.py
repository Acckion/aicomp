"""Register IR inputs and retain native training, with a rank-zero job lock."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys

import train_baseline as baseline
import zip_ir_dataset
import extra_iou_metrics
from mechanism_runtime import install_checkpoint_controls,install_training_controls
from src.core.yaml_utils import load_config

def main():
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT})
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',required=True)
    args,_=parser.parse_known_args();config=load_config(args.config)
    output=Path(config['output_dir']);lock=None
    if int(os.environ.get('RANK','0'))==0:
        lock=(baseline.ROOT/'experiments/mechanism_trials/ir_detector800.train.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'COMPLETE').exists():raise RuntimeError('Refusing to overwrite completed IR experiment')
    if (output/'metrics.jsonl').exists() and '--resume' not in sys.argv:
        raise RuntimeError('Partial IR experiment requires --resume')
    install_training_controls(baseline)
    install_checkpoint_controls(baseline,config)
    try:baseline.main()
    finally:
        if lock is not None:lock.close()

if __name__=='__main__':main()
