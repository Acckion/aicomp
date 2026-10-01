"""Runtime-only BN and progress controls; checkpoint parameter names stay intact."""
import copy
import fcntl
import json
from pathlib import Path
import re
import shutil
import time
import types

import torch


def install_checkpoint_controls(baseline, config):
    """Keep full best/last, compact snapshots at intervals; serialize disk writes."""
    original_save = baseline.atomic_save
    interval = int(config.get('inference_checkpoint_interval', 1))
    retain_epoch_states = config.get('save_epoch_training_states', True)
    inference_only = config.get('checkpoint_format') == 'ema_inference'
    selected_epochs = config.get('inference_checkpoint_epochs')
    lock_path = Path(baseline.ROOT) / 'experiments/mechanism_training/checkpoint_write.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(v) for v in value)
        return 0

    def save(state, path):
        path = Path(path)
        if re.fullmatch(r'epoch_\d+\.pth', path.name) and not retain_epoch_states:
            return
        compact = re.fullmatch(r'weights_epoch_(\d+)\.pth', path.name)
        if compact and selected_epochs is not None and int(compact.group(1)) not in selected_epochs:
            return
        if compact and int(compact.group(1)) % interval and int(compact.group(1)) != config['epochs']:
            if selected_epochs is None:
                return
        if inference_only and path.name in ('best.pth', 'last.pth'):
            weights = state['ema']['module'] if 'ema' in state else state['model']
            state = {'model': weights, 'last_epoch': state.get('last_epoch', -1),
                     'best_ap': state.get('best_ap'), 'best_epoch': state.get('best_epoch'),
                     'num_classes': config['num_classes'], 'source': 'EMA',
                     'checkpoint_format': 'ema_inference',
                     'note': 'Inference/init only; optimizer/RNG state is not retained.'}
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            required = tensor_bytes(state) + 16 * 1024**2 + 512 * 1024**2
            free = shutil.disk_usage(path.parent).free
            if free <= required:
                raise RuntimeError(f'Insufficient checkpoint disk space for {path}: '
                                   f'{free / 1024**3:.2f} GiB free, need >{required / 1024**3:.2f} GiB')
            original_save(state, path)

    baseline.atomic_save = save
    print(f'CHECKPOINT_POLICY best/last retained, full epoch states={retain_epoch_states}, '
          f'compact interval={interval}, inference-only best/last={inference_only}, '
          f'explicit snapshot epochs={selected_epochs}, serialized writes with >512 MiB reserve', flush=True)


def _train_with_frozen_statistics(self, mode=True):
    torch.nn.Module.train(self, mode)
    for module in self.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train(False)
    return self


def freeze_bn_statistics(model):
    """Freeze running buffers only, including after each outer model.train call."""
    model.train = types.MethodType(_train_with_frozen_statistics, model)
    model.train(model.training)
    return sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def install_training_controls(baseline):
    original_setup = baseline.BaselineSolver._setup
    original_train = baseline.train_one_epoch

    def setup(solver):
        original_setup(solver)
        frozen = solver.cfg.yaml_cfg.get('freeze_bn_statistics', False)
        if frozen:
            count = freeze_bn_statistics(solver.model)
            if solver.ema is not None:
                freeze_bn_statistics(solver.ema.module)
            print(f'Frozen BN running statistics in {count} modules; affine gradients retained', flush=True)

    def train(model, criterion, loader, optimizer, device, epoch, use_wandb, **kwargs):
        callback = kwargs.get('update_callback')
        output = kwargs.get('output_dir')
        updates = [0]
        latest_loss = [None]
        original_forward = criterion.forward

        def capture_loss(*args, **forward_kwargs):
            result = original_forward(*args, **forward_kwargs)
            latest_loss[0] = sum(result.values()).detach()
            return result

        criterion.forward = capture_loss

        def on_update(next_batch):
            updates[0] += 1
            if output is not None and (updates[0] <= 3 or updates[0] % 100 == 0):
                loss = float(latest_loss[0])
                if not torch.isfinite(latest_loss[0]).all():
                    raise RuntimeError(f'Non-finite startup loss: {loss}')
                path = Path(output) / 'optimizer_progress.json'
                state = {'epoch': epoch + 1, 'next_batch': next_batch,
                         'epoch_optimizer_updates': updates[0], 'time': time.time(),
                         'loss': loss,
                         'rank_pairs': getattr(criterion, 'last_rank_pairs', None),
                         'lr': [g['lr'] for g in optimizer.param_groups],
                         'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(state, indent=2))
                tmp.replace(path)
                if updates[0] == 1:
                    print('OPTIMIZER_PROGRESS ' + json.dumps(state), flush=True)
            if callback is not None:
                callback(next_batch)

        kwargs['update_callback'] = on_update
        try:
            return original_train(model, criterion, loader, optimizer, device, epoch, use_wandb, **kwargs)
        finally:
            criterion.forward = original_forward

    baseline.BaselineSolver._setup = setup
    baseline.train_one_epoch = train


def check_bn_behavior():
    """Real gradients and buffers: frozen stats, learnable affine, adaptive control."""
    torch.manual_seed(20260929)
    adaptive = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 1, 1))
    frozen = copy.deepcopy(adaptive)
    before_keys = tuple(frozen.state_dict())
    freeze_bn_statistics(frozen)
    before = frozen[0].running_mean.clone()
    x = torch.randn(4, 3, 8, 8) + 3
    frozen.eval().train()
    frozen(x).square().mean().backward()
    assert torch.equal(frozen[0].running_mean, before)
    assert int(frozen[0].num_batches_tracked) == 0
    assert frozen[0].weight.requires_grad and frozen[0].weight.grad.abs().sum() > 0
    assert before_keys == tuple(frozen.state_dict())
    adaptive.train()
    adaptive(x).square().mean().backward()
    assert not torch.equal(adaptive[0].running_mean, before)
    assert int(adaptive[0].num_batches_tracked) == 1
    cloned = copy.deepcopy(frozen)
    cloned.train()
    assert cloned.train.__self__ is cloned and not cloned[0].training
    print('BN NUMERICAL CHECK PASSED: frozen buffers, affine gradient, adaptive update, unchanged state keys', flush=True)
