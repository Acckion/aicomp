"""Separate IR spatial content from geometry/validity priors on fixed val400."""
import fcntl
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import ir_content_alignment
import extra_iou_metrics
import train_baseline as baseline
from src.core import YAMLConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'experiments/ir_information_probe'; out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'experiments/mechanism_trials/gpu6.lock').open('a')
    (out / 'status.json').write_text(json.dumps({'stage': 'waiting_for_spatial_audit'}))
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory)
    cfg = YAMLConfig(str(ROOT / 'configs/ir_content800.yml'))
    model = cfg.model.cuda().eval()
    checkpoint = ROOT / 'runs/ir_content800/best.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'], strict=True); del state
    mode = 'real'

    def intervene(module, inputs, output):
        features = list(output); feature = features[0]
        if mode == 'geometry_only':
            features[0] = torch.zeros_like(feature)
        elif mode == 'spatial_shuffled':
            generator = torch.Generator(device=feature.device).manual_seed(20261002)
            order = torch.randperm(feature.shape[-2] * feature.shape[-1], device=feature.device, generator=generator)
            features[0] = feature.flatten(2)[:, :, order].reshape_as(feature)
        elif mode == 'global_mean':
            valid = F.interpolate(model._valid.float(), size=feature.shape[-2:], mode='nearest')
            mean = (feature.float() * valid).sum((-2, -1), keepdim=True) / valid.sum((-2, -1), keepdim=True).clamp_min(1)
            features[0] = mean.to(feature.dtype).expand_as(feature)
        return features

    model.ir_encoder.register_forward_hook(intervene)
    results = {}
    for mode in ['real', 'geometry_only', 'spatial_shuffled', 'global_mean']:
        (out / 'status.json').write_text(json.dumps({'stage': 'evaluating', 'condition': mode}))
        stats, evaluator = baseline.evaluate(model, cfg.criterion.cuda(), cfg.postprocessor, cfg.val_dataloader,
                                             cfg.evaluator, 'cuda', 0, False)
        precision = evaluator.coco_eval['bbox'].eval['precision']
        stats['per_class_ap'] = {}
        for i, category in enumerate(evaluator.coco_eval['bbox'].params.catIds):
            values = precision[:, :, i, 0, -1]; values = values[values >= 0]
            stats['per_class_ap'][evaluator.coco_gt.cats[category]['name']] = float(values.mean()) if len(values) else None
        results[mode] = stats
        (out / 'report.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'validation_images': 400,
                                                   'results': results, 'unchanged': 'RGB, boxes, valid-IR mask, geometry embeddings and trained weights',
                                                   'limitations': 'These interventions are distribution shifts, not training ablations; they distinguish sensitivity, not guaranteed gains or phase2 transfer.'}, indent=2))
        print('IR_INFORMATION', mode, stats['coco_eval_bbox'][0] * 100, flush=True)
    (out / 'status.json').write_text(json.dumps({'stage': 'complete', 'conditions': list(results)}))


if __name__ == '__main__':
    main()
