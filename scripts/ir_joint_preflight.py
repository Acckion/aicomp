"""Check a mature-parent joint IR neck against an identical frozen control."""
import io
import json
import random
from copy import deepcopy
from pathlib import Path
import torch
import ir_joint_adaptation
from src.core import YAMLConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/ir_joint'


def main():
    torch.manual_seed(20261002); random.seed(20261002)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    cfg = YAMLConfig(str(ROOT / 'configs/ir_joint800.yml'))
    dataset = cfg.train_dataloader.dataset
    assert len(dataset) == 1600 and len(cfg.val_dataloader.dataset) == 400
    assert set(dataset.ids).isdisjoint(cfg.val_dataloader.dataset.ids)
    parent = ROOT / 'runs/ir_content800/best.pth'
    ckpt = torch.load(parent, map_location='cpu', weights_only=False)
    weights = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    model = cfg.model.cuda(); model.load_state_dict(weights, strict=True)
    del weights, ckpt
    samples, targets = [], []
    for i in [10, 11]:
        sample, target = dataset[i]; samples.append(sample)
        targets.append({k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in target.items()})
    x = torch.stack(samples).cuda(); model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        joint = model(x)
        model.adapt_ir_neck = False; frozen = model(x); model.adapt_ir_neck = True
        assert torch.equal(joint['pred_boxes'], frozen['pred_boxes'])
        assert torch.equal(joint['pred_logits'], frozen['pred_logits'])
        del joint, frozen
        ema = deepcopy(model)
        for layer in ema.decoder.decoder.layers[-3:]:
            assert list(layer._forward_pre_hooks.values())[-1].__self__ is ema
        del ema
    criterion = cfg.criterion.cuda(); optimizer = cfg.optimizer
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    grouped = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(grouped) == len(set(grouped)) and set(grouped) == trainable
    from mechanism_runtime import freeze_bn_statistics
    freeze_bn_statistics(model); model.train()
    x = torch.nn.functional.interpolate(x, size=(992, 992))
    scaler = torch.amp.GradScaler('cuda', init_scale=1)
    records = []
    for step in range(3):
        t = deepcopy(targets)
        if step == 2:
            x[:, 6] = 0  # Missing modality together with empty GT.
            for target in t:
                for key in ['boxes', 'labels', 'area', 'iscrowd']:
                    if key in target: target[key] = target[key][:0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = sum(criterion(model(x, t), t, epoch=0).values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, bad
        neck_grad = sum(float(p.grad.float().norm()) for p in model.ir_encoder.parameters() if p.grad is not None)
        if step < 2: assert neck_grad > 0
        assert all(p.grad is None for p in model.ir_backbone.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        scaler.step(optimizer); scaler.update()
        records.append({'step': step, 'loss': float(loss), 'ir_neck_grad_norm_sum': neck_grad})
        print('IR_JOINT_PREFLIGHT_STEP', records[-1], flush=True)
    model.eval(); model.ir_mode = 'zero'
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        test = torch.stack(samples).cuda(); model.ir_enabled = False; a = model(test)
        model.ir_enabled = True; b = model(test)
        assert torch.equal(a['pred_boxes'], b['pred_boxes']) and torch.equal(a['pred_logits'], b['pred_logits'])
        del a, b
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
    model.adapt_ir_neck = False; model.ir_encoder.requires_grad_(False)
    model.ir_mode = 'paired'; model.train(); optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = sum(criterion(model(torch.stack(samples).cuda(), targets), targets, epoch=0).values())
    loss.backward()
    assert all(p.grad is None for p in model.ir_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    report = {'stage': 'passed', 'parent': str(parent), 'same_parent_outputs_exact': True,
              'train_val_disjoint': True, 'ema_hook_owner_correct': True,
              'strict_reload_passed': True, 'zero_ir_exact': True,
              'control_neck_gradient_absent': True, 'frozen_backbone_gradient_absent': True,
              'max_scale': 992, 'batch_size': 2, 'optimizer_steps': records,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'preflight.json').write_text(json.dumps(report, indent=2))
    print('IR_JOINT_PREFLIGHT_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__': main()
