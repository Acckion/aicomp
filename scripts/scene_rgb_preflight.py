"""Fresh public-pretrained grouped baseline: maximum scale and true updates."""
from copy import deepcopy
import io
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import train_baseline as baseline
from src.core import YAMLConfig
from mechanism_runtime import install_training_controls

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    cfg = YAMLConfig(str(ROOT / 'configs/scene_rgb800.yml'))
    cfg.tuning = str(ROOT / 'checkpoints/dfine_x_obj2coco.pth')
    install_training_controls(baseline)
    solver = baseline.BaselineSolver(cfg)
    solver._setup()
    loader = cfg.train_dataloader
    val = cfg.val_dataloader
    assert len(loader.dataset) == 1610 and len(val.dataset) == 390
    train_ids, val_ids = set(loader.dataset.ids), set(val.dataset.ids)
    assert train_ids.isdisjoint(val_ids)
    report = json.loads((ROOT / 'experiments/scene_groups/report.json').read_text())
    assert all(not (set(g) & train_ids and set(g) & val_ids) for g in report['groups'])
    x, targets = next(iter(loader))
    x = F.interpolate(x.cuda(), size=(992, 992), mode='bilinear', align_corners=False)
    targets = [{k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in t.items()} for t in targets]
    optimizer = cfg.optimizer
    model = solver.model.train()
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    records = []
    updates = []
    hook = optimizer.register_step_post_hook(lambda *args: updates.append(True))
    for step in range(3):
        t = deepcopy(targets)
        if step == 2:
            for target in t:
                for k in ('boxes', 'labels', 'area', 'iscrowd'):
                    if k in target:
                        target[k] = target[k][:0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = sum(solver.criterion(model(x, t), t, epoch=0).values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        scaler.step(optimizer)
        scaler.update()
        solver.ema.update(model)
        record = {'step': step + 1, 'loss': float(loss.detach()), 'empty_gt': step == 2}
        records.append(record)
        print(json.dumps(record), flush=True)
    assert len(updates) == 3
    hook.remove()
    solver.ema.module.eval()
    stream = io.BytesIO()
    torch.save(solver.ema.module.state_dict(), stream)
    stream.seek(0)
    solver.ema.module.load_state_dict(torch.load(stream, map_location='cpu', weights_only=True), strict=True)
    vx, _ = next(iter(val))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        result = solver.ema.module(vx.cuda())
    assert torch.isfinite(result['pred_logits']).all() and torch.isfinite(result['pred_boxes']).all()
    output = ROOT / 'experiments/scene_rgb_gpu6'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preflight.json').write_text(json.dumps({'stage':'passed','actual_updates':len(updates),
        'source':'public dfine_x_obj2coco only; no prior competition checkpoint', 'records':records,
        'training_images':1610,'heldout_images':390,'maximum_size':992,'ema_resident':True,
        'group_disjoint':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2},indent=2))


if __name__ == '__main__':
    main()
