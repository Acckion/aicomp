"""Inspect trained IR offsets and intermediate RGB candidate coverage on train only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import torch
from torchvision.ops import box_iou
import ir_content_alignment
from ir_query_alignment import PairedIRCoco
from src.core import YAMLConfig

ROOT = Path(__file__).resolve().parents[1]


def xyxy(boxes):
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=128)
    args = parser.parse_args()
    out = ROOT / 'experiments/ir_spatial_audit'
    out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'experiments/mechanism_trials/gpu6.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2); torch.manual_seed(20261002); random.seed(20261002)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory)
    cfg = YAMLConfig(str(ROOT / 'configs/ir_content800.yml'))
    model = cfg.model.cuda().eval()
    checkpoint = ROOT / 'runs/ir_content800/best.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'], strict=True)
    del state
    dataset = PairedIRCoco(str(ROOT / 'data/train'), str(ROOT / 'data/annotations/train1600.json'), None,
                          cfg.yaml_cfg['val_dataloader']['dataset']['archive'], training=False, size=800)
    records = []
    candidates = {}
    current = {}

    def inspect(layer):
        def hook(module, inputs):
            q, refs, features, valid = inputs
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                box = refs[:, :, 0].float()
                radius = (box[..., 2:] * .5).clamp(.025, .12)
                positions = box[..., :2].unsqueeze(2) + module.grid[None, None] * radius.unsqueeze(2)
                values, mask = module.sample(features, valid, positions)
                query = module.query(q.float()).unsqueeze(2); keys = module.key(values)
                weights = ((query * keys).sum(-1) / 8).masked_fill(~mask, -1e4).softmax(-1) * mask
                weights /= weights.sum(-1, keepdim=True).clamp_min(1e-6)
                shift = module.shift(((keys * query) * weights.unsqueeze(-1)).sum(2)).tanh() * .06
                records.append({'image_id': current['image_id'], 'layer': layer,
                                'mean_abs_shift': float(shift.abs().mean()),
                                'p95_abs_shift': float(torch.quantile(shift.abs().flatten(), .95)),
                                'saturated_fraction': float((shift.abs() >= .054).float().mean()),
                                'valid_fraction': float(mask.float().mean()),
                                'min_radius_fraction': float((radius <= .025).float().mean())})
                candidates[layer] = box[0].detach().cpu()
        return hook

    for layer, sampler in enumerate(model.ir_samplers):
        sampler.register_forward_pre_hook(inspect(layer))
    coverage = []
    indexes = torch.linspace(0, len(dataset) - 1, min(args.images, len(dataset))).long().tolist()
    for index in indexes:
        image, target = dataset[index]
        current['image_id'] = int(target['image_id'].item())
        candidates.clear()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            result = model(image.unsqueeze(0).cuda())
        gt = target['boxes'].as_subclass(torch.Tensor).float()
        # This repository stores orig_size as [width, height], unlike many
        # torchvision examples. Check against image metadata explicitly.
        width, height = target['orig_size'].float()
        meta = dataset.coco.loadImgs([current['image_id']])[0]
        assert int(width) == meta['width'] and int(height) == meta['height']
        gt = gt / torch.tensor([width, height, width, height])
        assert not len(gt) or ((gt >= 0).all() and (gt <= 1.00001).all())
        final = xyxy(result['pred_boxes'][0].float().cpu())
        # Geometry-only coverage diagnoses candidate support; no class/score
        # gate, AP or per-GT oracle predictions are used as training targets.
        row = {'image_id': current['image_id'], 'gt': len(gt)}
        for label, boxes in [('before_ir', xyxy(candidates[0])), ('final', final)]:
            best = box_iou(gt, boxes).amax(1) if len(gt) else torch.empty(0)
            row[label] = {str(t): int((best >= t).sum()) for t in [.3, .5, .75, .9]}
        coverage.append(row)
        if len(coverage) % 16 == 0:
            print('AUDIT', len(coverage), 'training images', flush=True)
        (out / 'status.json').write_text(json.dumps({'stage': 'running', 'images': len(coverage), 'target': len(indexes), 'pid': os.getpid()}))
    totals = {'gt': sum(r['gt'] for r in coverage)}
    for label in ['before_ir', 'final']:
        totals[label] = {str(t): sum(r[label][str(t)] for r in coverage) for t in [.3, .5, .75, .9]}
    summary = {}
    for layer in range(3):
        subset = [r for r in records if r['layer'] == layer]
        summary[str(layer)] = {key: sum(r[key] for r in subset) / len(subset) for key in
                              ['mean_abs_shift', 'p95_abs_shift', 'saturated_fraction', 'valid_fraction', 'min_radius_fraction']}
    report = {'stage': 'complete', 'checkpoint': str(checkpoint), 'training_images_only': len(indexes),
              'candidate_geometry': totals, 'layers': summary, 'per_image': coverage, 'sampler_records': records,
              'limitations': 'Intermediate boxes can move through later decoder layers; low geometry coverage is not proof of unrecoverable misses. Image-averaged offset summaries include background queries. No training or validation fitting.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'status.json').write_text(json.dumps({'stage': 'complete', 'images': len(coverage)}))
    print(json.dumps({'candidate_geometry': totals, 'layers': summary}), flush=True)


if __name__ == '__main__':
    main()
