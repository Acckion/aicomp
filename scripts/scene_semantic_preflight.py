"""Fresh public-pretrained grouped baseline: maximum scale and true updates."""
from copy import deepcopy
import io
import json
from pathlib import Path
import torch
import argparse
import semantic_init_rows
import torch.nn.functional as F
import train_baseline as baseline
from src.core import YAMLConfig
from mechanism_runtime import install_training_controls

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['scene_semantic_init800','scene_rgb_fastcontrol800','scene_obj365_reset800','scene_obj365_pool800','full2000_obj365_pool800','full2000_obj365_reset800','full2000_obj365_pool800_b2','full2000_obj365_reset800_b2'],required=True);args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    cfg = YAMLConfig(str(ROOT / 'configs' / (args.name+'.yml')))
    limit=cfg.yaml_cfg.get('gpu_memory_limit_gib',6)
    assert 0<limit<=8.5
    torch.cuda.set_per_process_memory_fraction(limit * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    full = args.name.startswith('full2000_')
    pooled = '_obj365_pool800' in args.name
    if pooled:
        import taxonomy_pooling
    source = 'dfine_x_obj365.pth' if '_obj365_' in args.name else 'dfine_x_obj2coco.pth'
    cfg.tuning = str(ROOT / 'checkpoints' / source)
    install_training_controls(baseline)
    solver = baseline.BaselineSolver(cfg)
    solver._setup()
    loader = cfg.train_dataloader
    val = None if full else cfg.val_dataloader
    train_ids = set(loader.dataset.ids)
    if full:
        assert len(loader.dataset) == 2000 and cfg.yaml_cfg['validate'] is False
        shadow_train = json.loads((ROOT/'data/annotations/scene_train.json').read_text())
        shadow_val = json.loads((ROOT/'data/annotations/scene_val.json').read_text())
        assert train_ids == {i['id'] for i in shadow_train['images']+shadow_val['images']}
    else:
        assert len(loader.dataset) == 1610 and len(val.dataset) == 390
        val_ids = set(val.dataset.ids)
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
        if pooled and step < 2:
            head = model.decoder.enc_score_head
            assert head.source.weight.grad is not None and head.source.weight.grad.abs().sum() > 0
            assert head.residual.weight.grad is not None and head.residual.weight.grad.abs().sum() > 0
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
    vx = F.interpolate(x[:1],size=(800,800),mode='bilinear',align_corners=False) if full else next(iter(val))[0]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        result = solver.ema.module(vx.cuda())
    assert torch.isfinite(result['pred_logits']).all() and torch.isfinite(result['pred_boxes']).all()
    output = ROOT / 'experiments/scene_semantic_init' / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preflight.json').write_text(json.dumps({'stage':'passed','actual_updates':len(updates),
        'semantic_mapping':getattr(solver,'semantic_initialization',None),'source':'public '+source+' only; no prior competition checkpoint', 'records':records,
        'training_images':len(loader.dataset),'heldout_images':0 if full else 390,'maximum_size':992,'ema_resident':True,
        'batch_size':cfg.yaml_cfg['train_dataloader']['total_batch_size'],
        'group_disjoint':None if full else True,'validation_disabled':full,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2},indent=2))


if __name__ == '__main__':
    main()
