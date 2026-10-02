"""Train-only full-frame800 versus1920x1088 geometry; no target crops."""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'D-FINE'), str(ROOT / 'scripts')]
import torch
import fcntl
import native_grid
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert, box_iou
from src.core import YAMLConfig


def main():
    lock=(ROOT/"experiments/mechanism_trials/gpu2.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = YAMLConfig(str(ROOT / 'configs/ft_aug800_shared3.yml'))
    model = cfg.model
    model.load_state_dict(native_grid.parent_weights(ROOT / 'runs/ft_aug800/weights_epoch_020.pth'), strict=True)
    model.encoder.eval_spatial_size=None;model.decoder.eval_spatial_size=None
    model.cuda().eval()
    data = json.loads((ROOT / 'data/annotations/train1600.json').read_text())
    images = {i['id']: i for i in data['images']}
    candidates = [a for a in data['annotations'] if images[a['image_id']]['width']>=1280 and min(a['bbox'][2] * 800 / images[a['image_id']]['width'],
                                                      a['bbox'][3] * 800 / images[a['image_id']]['height']) < 16]
    random.Random(20260929).shuffle(candidates)
    chosen, seen = [], set()
    for a in candidates:
        if a['image_id'] not in seen:
            chosen.append(a)
            seen.add(a['image_id'])
        if len(chosen) == 64:
            break
    records = []

    def measure(image, truth, category, size=(800,800)):
        with torch.autocast("cuda",dtype=torch.float16):
            output = model(TF.to_tensor(TF.resize(image, list(size))).unsqueeze(0).cuda())
        boxes = box_convert(output['pred_boxes'][0].float(), 'cxcywh', 'xyxy').clamp(0, 1)
        scores = output['pred_logits'][0].sigmoid()
        iou = box_iou(boxes, truth[None]).flatten()
        correct = scores.argmax(-1) == category
        return {'best_any_iou': float(iou.max()),
                'best_argmax_class_iou': float(iou[correct].max()) if correct.any() else 0.0}

    with torch.inference_mode():
        for a in chosen:
            im = images[a['image_id']]
            image = Image.open(ROOT / 'data/train' / im['file_name']).convert('RGB')
            x, y, w, h = a['bbox']
            truth = torch.tensor([x, y, x + w, y + h], dtype=torch.float32, device='cuda')
            whole = measure(image, truth / truth.new_tensor([im['width'], im['height']] * 2), a['category_id'])
            local = measure(image, truth / truth.new_tensor([im['width'], im['height']] * 2), a['category_id'], size=(1088,1920))
            records.append({'image_id': im['id'], 'annotation_id': a['id'], 'category_id': a['category_id'], 'whole': whole, 'native': local})
    report = {'training_images': len(records), 'checkpoint': 'ft_aug800 epoch20 EMA',
              'full_input_size': 800, 'native_input_size': [1088,1920], 'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
              'notes': ['Seen train1600 images, one target per image sampled from short-side<16 at baseline800.',
                        'Full frame and all pixels; no target-aware crops. Train-only optimistic geometry, not AP or validation improvement.',
                        'Maximum IoU geometry is optimistic, not one-to-one recall/AP; frozen model, no gradients.'], 'summary': {}}
    for key in ['best_any_iou', 'best_argmax_class_iou']:
        whole = torch.tensor([r['whole'][key] for r in records])
        zoom = torch.tensor([r['native'][key] for r in records])
        report['summary'][key] = {'whole_mean': float(whole.mean()), 'native_mean': float(zoom.mean()),
            'native_better_by_over005': int((zoom > whole + .05).sum()),
            'native_worse_by_over005': int((zoom < whole - .05).sum()),
            'whole_iou75': int((whole >= .75).sum()), 'native_iou75': int((zoom >= .75).sum()),
            'whole_iou90': int((whole >= .9).sum()), 'native_iou90': int((zoom >= .9).sum())}
    out = ROOT / 'experiments/native_fullframe_probe'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'records.json').write_text(json.dumps(records))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
