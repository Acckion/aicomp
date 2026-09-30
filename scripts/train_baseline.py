"""AICOMP RGB training using official D-FINE model, losses and augmentation.

Unlike the upstream solver, this runner never rewinds the epoch/optimizer when
strong augmentation stops. Full-data training has no validation or best-AP claim.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
import numpy as np
import torch
import torch.distributed as dist
import yaml
from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver._solver import BaseSolver
from src.solver.det_engine import train_one_epoch, evaluate
from schedule import WarmupCosine
import target_views  # register train-only augmentation before YAML construction


def atomic_save(state, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


class BaselineSolver(BaseSolver):
    def load_tuning_state(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        weights = state['ema']['module'] if 'ema' in state else state['model']
        current = self.model.state_dict()
        # Resolution-dependent anchors are generated from the new config.
        weights = dict(weights)
        for key in ('decoder.anchors', 'decoder.valid_mask'):
            if key in current:
                weights[key] = current[key]
        matched, info = self._matched_state(current, weights)
        # AICOMP's 12 classes do not follow COCO/Objects365's class ordering.
        # Shape-mismatched classification layers are intentionally reinitialized.
        if any('score_head' not in k and 'denoising_class_embed' not in k
               for k in info['missed'] + info['unmatched']):
            raise RuntimeError(f'Unexpected pretrained architecture mismatch: {info}')
        self.model.load_state_dict(matched, strict=False)
        print(f'Pretrained tensors loaded: {len(matched)}; reinitialized: {info}', flush=True)


class MigrationRequested(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--init-checkpoint', help='initialize model/EMA only; fresh optimizer and schedule')
    parser.add_argument('--evaluate-init', action='store_true')
    parser.add_argument('--rebatch-resume', action='store_true', help='rebase schedule for new batch/world size; replay unfinished epoch')
    parser.add_argument('--smoke', action='store_true', help='one train and validation batch, no weights saved')
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing to silently train on CPU')
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    dist_utils.setup_print(rank == 0)
    dist_utils.setup_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = YAMLConfig(args.config)
    if args.resume and args.init_checkpoint:
        raise ValueError('resume and init-checkpoint are mutually exclusive')
    cfg.tuning = None if args.resume else (args.init_checkpoint or str(ROOT / 'checkpoints/dfine_x_obj2coco.pth'))
    cfg.resume = None
    if cfg.yaml_cfg.get('gpu_memory_limit_gib'):
        limit_bytes = cfg.yaml_cfg['gpu_memory_limit_gib'] * 1024**3
        total_bytes = torch.cuda.get_device_properties(local_rank).total_memory
        torch.cuda.set_per_process_memory_fraction(limit_bytes / total_bytes, local_rank)
    solver = BaselineSolver(cfg)
    solver._setup()
    solver.optimizer = cfg.optimizer
    solver.train_dataloader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)
    do_eval = cfg.yaml_cfg.get('validate', True)
    if do_eval:
        solver.val_dataloader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
        solver.evaluator = cfg.evaluator
    accumulation = cfg.yaml_cfg.get('gradient_accumulation_steps', 1)
    steps_per_epoch = (len(solver.train_dataloader) + accumulation - 1) // accumulation
    solver.lr_warmup_scheduler = WarmupCosine(
        solver.optimizer, cfg.epochs * steps_per_epoch,
        cfg.yaml_cfg.get('warmup_epochs', 3) * steps_per_epoch,
        cfg.yaml_cfg.get('min_lr_factor', .01))
    new_schedule = solver.lr_warmup_scheduler.state_dict()
    best_ap, best_epoch = -1., None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        solver.load_state_dict(checkpoint)
        if args.rebatch_resume:
            old_schedule = checkpoint['lr_warmup_scheduler']
            fraction = old_schedule['step_index'] / old_schedule['total_steps']
            new_schedule['step_index'] = round(fraction * new_schedule['total_steps'])
            solver.lr_warmup_scheduler.load_state_dict(new_schedule)
            print('Rebased LR schedule; unfinished epoch will be replayed with new sharding.', flush=True)
        best_ap = checkpoint.get('best_ap', -1.)
        best_epoch = checkpoint.get('best_epoch')
        rng = checkpoint.get('rng_by_rank', [])
        if rank < len(rng):
            random.setstate(rng[rank]['python'])
            np.random.set_state(rng[rank]['numpy'])
            torch.set_rng_state(rng[rank]['torch'])
            torch.cuda.set_rng_state(rng[rank]['cuda'])
    output = Path(cfg.output_dir)
    if rank == 0:
        (output / 'resolved_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg, sort_keys=False))
        print(f'world_size={world}, steps/epoch={steps_per_epoch}, epochs={cfg.epochs}, validation={do_eval}', flush=True)
    if args.smoke:
        solver.train_dataloader.set_epoch(0)
        # Exercise peak multi-scale memory and two DDP steps (unused parameters
        # can otherwise only surface after the first backward completes).
        peak = max(solver.train_dataloader.collate_fn.scales or [cfg.yaml_cfg.get('eval_spatial_size', [640])[0]])
        solver.train_dataloader.collate_fn.scales = [peak]
        batches = iter(solver.train_dataloader)
        smoke_batches = [next(batches) for _ in range(max(2, accumulation))]
        train_one_epoch(solver.model, solver.criterion, smoke_batches,
                        solver.optimizer, solver.device, 0, False, max_norm=cfg.clip_max_norm,
                        ema=solver.ema, scaler=solver.scaler, print_freq=1,
                        gradient_accumulation_steps=accumulation)
        if do_eval:
            evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
                     [next(iter(solver.val_dataloader))], solver.evaluator, solver.device, 0, False)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        print('SMOKE TEST PASSED', flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    if args.evaluate_init and do_eval and not args.resume:
        initial, _ = evaluate(solver.ema.module if solver.ema else solver.model,
                              solver.criterion, solver.postprocessor, solver.val_dataloader,
                              solver.evaluator, solver.device, -1, False)
        best_ap = initial['coco_eval_bbox'][0]
        best_epoch = 0
        if rank == 0:
            (output / 'initial_metrics.json').write_text(json.dumps(initial, indent=2))
            initial_state = solver.state_dict()
            initial_state.update(best_ap=best_ap, best_epoch=0)
            atomic_save(initial_state, output / 'best.pth')
    requested = [False]
    signal.signal(signal.SIGUSR1, lambda *unused: requested.__setitem__(0, True))
    def migration_callback(next_batch):
        if not requested[0]:
            return
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        state = solver.state_dict()
        state.update(best_ap=best_ap, best_epoch=best_epoch, rng_by_rank=[rng],
                     migration={'unfinished_epoch': solver.last_epoch + 1, 'next_batch': next_batch,
                                'note': 'Replay unfinished epoch when changing world size; not bitwise equivalent.'})
        if world != 1:
            raise RuntimeError('Migration signal currently supports single-GPU source jobs only')
        atomic_save(state, output / 'migration.pth')
        (output / 'MIGRATION_READY').write_text('Saved model, EMA, optimizer, scaler and schedule at optimizer boundary.\n')
        raise MigrationRequested()
    for epoch in range(solver.last_epoch + 1, cfg.epochs):
        started = time.monotonic()
        solver.train_dataloader.set_epoch(epoch)
        if world > 1:
            solver.train_dataloader.sampler.set_epoch(epoch)
        try:
            stats = train_one_epoch(
                solver.model, solver.criterion, solver.train_dataloader, solver.optimizer,
                solver.device, epoch, False, epochs=cfg.epochs, max_norm=cfg.clip_max_norm,
                print_freq=cfg.print_freq, ema=solver.ema, scaler=solver.scaler,
                gradient_accumulation_steps=accumulation, update_callback=migration_callback,
                lr_warmup_scheduler=solver.lr_warmup_scheduler, writer=solver.writer,
                output_dir=output)
        except MigrationRequested:
            print("MIGRATION CHECKPOINT SAVED", flush=True)
            return
        solver.last_epoch = epoch
        metrics, improved = {}, False
        if do_eval:
            metrics, evaluator = evaluate(
                solver.ema.module if solver.ema else solver.model, solver.criterion,
                solver.postprocessor, solver.val_dataloader, solver.evaluator,
                solver.device, epoch, False)
            ap = metrics['coco_eval_bbox'][0]
            if not np.isfinite(ap):
                raise RuntimeError(f'Non-finite AP: {ap}')
            improved = ap > best_ap
            if improved:
                best_ap, best_epoch = ap, epoch + 1
            if rank == 0:
                precisions = evaluator.coco_eval['bbox'].eval['precision']
                per_class = {}
                for i, cat in enumerate(solver.val_dataloader.dataset.categories):
                    values = precisions[:, :, i, 0, -1]
                    values = values[values >= 0]
                    per_class[cat['name']] = float(values.mean()) if len(values) else None
                metrics['per_class_ap'] = per_class
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng[0] = rng
        if rank == 0:
            state = solver.state_dict()
            state.update(best_ap=best_ap, best_epoch=best_epoch, rng_by_rank=all_rng)
            atomic_save(state, output / 'last.pth')
            # Keep a compact inference snapshot from every epoch for later selection.
            inference_model = solver.ema.module if solver.ema else dist_utils.de_parallel(solver.model)
            atomic_save({'model': inference_model.state_dict(), 'last_epoch': epoch,
                         'num_classes': cfg.yaml_cfg['num_classes'], 'source': 'EMA' if solver.ema else 'model'},
                        output / f'weights_epoch_{epoch + 1:03d}.pth')
            if improved:
                atomic_save(state, output / 'best.pth')
            if (epoch + 1) % cfg.checkpoint_freq == 0 or epoch + 1 == cfg.epochs:
                atomic_save(state, output / f'epoch_{epoch + 1:03d}.pth')
            record = {'epoch': epoch + 1, 'train': stats, 'validation': metrics,
                      'best_ap': best_ap if do_eval else None, 'best_epoch': best_epoch,
                      'seconds': time.monotonic() - started,
                      'lr': [group['lr'] for group in solver.optimizer.param_groups]}
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            (output / 'status.json').write_text(json.dumps(record, indent=2))
            print(json.dumps(record), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        (output / 'COMPLETE').write_text(f'{cfg.epochs} epochs complete\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
