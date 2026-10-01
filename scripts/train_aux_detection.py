"""Isolated runner permitting auxiliary parameters and exporting native weights."""
import json
from pathlib import Path
import torch
import train_baseline as baseline
import aux_detection
import extra_iou_metrics
from mechanism_runtime import install_training_controls, install_checkpoint_controls
from src.core.yaml_utils import load_config
import argparse


def load_tuning(solver, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model'])
    current = solver.model.state_dict()
    for key in ('decoder.anchors', 'decoder.valid_mask'):
        if key in current: source[key] = current[key]
    unexpected = set(source) - set(current)
    missing = set(current) - set(source)
    if unexpected or any(not key.startswith('aux_head.') for key in missing):
        raise RuntimeError(f'Parent architecture mismatch: missing={missing}, unexpected={unexpected}')
    if any(source[key].shape != current[key].shape for key in source):
        raise RuntimeError('Parent tensor shapes mismatch')
    solver.model.load_state_dict(source, strict=False)
    # The new head consumes RNG during construction; reset both arms before
    # data-loader shuffling/augmentation so its initialization does not alter them.
    baseline.dist_utils.setup_seed(20260929)
    print(f'AUX_PARENT loaded={len(source)}, new_head_parameters={len(missing)}', flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    baseline.BaselineSolver.load_tuning_state = load_tuning
    install_training_controls(baseline)
    install_checkpoint_controls(baseline, config)
    original_save = baseline.atomic_save
    def save(state, path):
        if Path(path).name.startswith('weights_epoch_'):
            state = dict(state)
            state['model'] = {key: value for key, value in state['model'].items()
                              if not key.startswith('aux_head.')}
            state['auxiliary_head_removed'] = True
        original_save(state, path)
    baseline.atomic_save = save
    original_train = baseline.train_one_epoch
    def train(model, criterion, loader, optimizer, device, epoch, use_wandb, **kwargs):
        callback = kwargs.get('update_callback')
        output = Path(config['output_dir'])
        updates = [0]
        def progress(batch):
            updates[0] += 1
            unwrapped = model.module if hasattr(model, 'module') else model
            if hasattr(unwrapped, 'aux_head') and (updates[0] <= 3 or updates[0] % 50 == 0):
                state = {'epoch': epoch+1, 'optimizer_updates': updates[0],
                         **unwrapped.aux_head.last_statistics}
                (output / 'aux_assignment.json').write_text(json.dumps(state, indent=2))
                with (output / 'aux_assignment.jsonl').open('a') as stream:
                    stream.write(json.dumps(state)+'\n')
                if updates[0] <= 3: print('AUX_ASSIGNMENT '+json.dumps(state), flush=True)
            if callback: callback(batch)
        kwargs['update_callback'] = progress
        return original_train(model, criterion, loader, optimizer, device, epoch, use_wandb, **kwargs)
    baseline.train_one_epoch = train
    try:
        baseline.main()
    finally:
        if torch.cuda.is_initialized():
            print(f'CUDA_PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated()/1024**2:.2f}', flush=True)


if __name__ == '__main__': main()
