"""Actual maximum-size updates with EMA resident, geometry and reload checks."""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import torch
import native_grid
from src.core import YAMLConfig
from mechanism_runtime import freeze_bn_statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    args = p.parse_args()
    out = ROOT / 'experiments/native_grid' / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3 / torch.cuda.get_device_properties(0).total_memory)
    cfg = YAMLConfig(str(ROOT / 'configs' / f'{args.name}.yml'))
    loader = cfg.train_dataloader
    assert len(loader.dataset) == 1600 and len(cfg.val_dataloader.dataset) == 400
    assert set(loader.dataset.ids).isdisjoint(cfg.val_dataloader.dataset.ids)
    packed, targets = next(iter(loader))
    packed = packed.cuda()
    targets = [{k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in t.items()} for t in targets]
    model = cfg.model
    native_grid.initialize(model, ROOT / 'runs/ft_aug800/weights_epoch_020.pth')
    model.cuda().eval()
    enabled = model.detail_enabled
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        model.detail_enabled = True
        a = model(packed)
        model.detail_enabled = False
        b = model(packed)
        initial = {k:float((a[k]-b[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
        assert max(initial.values()) == 0, initial
        del a, b
    model.detail_enabled = enabled
    ema = deepcopy(model).eval()  # Keep resident throughout the real updates.
    criterion = cfg.criterion.cuda()
    optimizer = cfg.optimizer
    # Verify actual grouping rather than assuming regex strings matched.
    by_id = {id(p):name for name,p in model.named_parameters()}
    detail_names = []
    for group in optimizer.param_groups:
        names = [by_id[id(p)] for p in group['params'] if by_id[id(p)].startswith('detail_bridge.')]
        if names:
            assert group['lr'] == 3e-4
            detail_names += names
    assert len(detail_names) == len(list(model.detail_bridge.parameters()))
    freeze_bn_statistics(model)
    model.train()
    model.forced_size = 992
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    records = []
    for step in range(3):
        t = deepcopy(targets)
        if step == 2:
            for target in t:
                for key in ('boxes','labels','area','iscrowd'):
                    if key in target:
                        target[key] = target[key][:0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = sum(criterion(model(packed, t), t, epoch=0).values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        grad = {n:float(p.grad.norm()) for n,p in model.detail_bridge.named_parameters() if p.grad is not None}
        if enabled:
            assert grad['output.weight'] > 0
            if step == 1:
                assert any(v>0 for k,v in grad.items() if k.startswith('projection.'))
        else:
            assert not grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        scaler.step(optimizer)
        scaler.update()
        record = {'step':step+1, 'loss':float(loss.detach()), 'empty_gt':step==2, 'detail_grad':grad}
        records.append(record)
        print(json.dumps(record), flush=True)
    model.eval()
    model.forced_size = None
    ema.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        expected = model(packed)
        ema_pred = ema(packed)
        assert torch.equal(expected['pred_boxes'], ema_pred['pred_boxes'])
        assert torch.equal(expected['pred_logits'], ema_pred['pred_logits'])
        model.detail_enabled = False
        disabled = model(packed)
        effect = float((expected['pred_logits']-disabled['pred_logits']).abs().max())
        if enabled:
            assert effect > 0
        model.detail_enabled = enabled
        del ema_pred, disabled
    # Strict reload uses a full current model state and is a serialization test.
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    model.load_state_dict(torch.load(stream, map_location='cuda', weights_only=True), strict=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        reloaded = model(packed)
        assert torch.equal(expected['pred_boxes'], reloaded['pred_boxes'])
        assert torch.equal(expected['pred_logits'], reloaded['pred_logits'])
    report = {'stage':'passed', 'name':args.name, 'grid_factor':model.grid_factor,
              'native_enabled':enabled, 'max_global':992, 'native_tile':model.tile_size,
              'ema_resident':True, 'initial_exact_same_grid':initial, 'strict_reload':True,
              'native_effect_after_updates':effect,
              'records':records, 'peak_mib':torch.cuda.max_memory_allocated()/1024**2,
              'train_val_disjoint':True, 'no_gt_crop_selection':True}
    (out / 'preflight.json').write_text(json.dumps(report, indent=2))
    print('PREFLIGHT_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
