"""Measure target-aware zoom teacher geometry on TRAINING images only."""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'D-FINE'), str(ROOT / 'scripts')]
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert, box_iou
from src.core import YAMLConfig


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = YAMLConfig(str(ROOT / 'configs/ft_aug800_shared3.yml'))
    model = cfg.model
    model.load_state_dict(torch.load(ROOT / 'runs/ft_aug800/weights_epoch_020.pth', map_location='cpu', weights_only=False)['model'], strict=True)
    model.cuda().eval()
    data = json.loads((ROOT / 'data/annotations/train1600.json').read_text())
    images = {i['id']: i for i in data['images']}
    candidates = [a for a in data['annotations'] if min(a['bbox'][2] * 800 / images[a['image_id']]['width'],
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

    def measure(image, truth, category):
        output = model(TF.to_tensor(TF.resize(image, [800, 800])).unsqueeze(0).cuda())
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
            # Context at least 35% of each image dimension; anchor fully retained.
            cw, ch = min(im['width'], max(round(im['width'] * .35), int(w + 2))), min(im['height'], max(round(im['height'] * .35), int(h + 2)))
            left = max(0, min(im['width'] - cw, round(x + w / 2 - cw / 2)))
            top = max(0, min(im['height'] - ch, round(y + h / 2 - ch / 2)))
            local_truth = (truth - truth.new_tensor([left, top] * 2)) / truth.new_tensor([cw, ch] * 2)
            local = measure(image.crop((left, top, left + cw, top + ch)), local_truth, a['category_id'])
            records.append({'image_id': im['id'], 'annotation_id': a['id'], 'category_id': a['category_id'], 'whole': whole, 'zoom': local})
    report = {'training_images': len(records), 'checkpoint': 'ft_aug800 epoch20 EMA',
              'full_input_size': 800, 'local_input_size': 800, 'context_fraction': .35,
              'notes': ['Seen train1600 images, one target per image sampled from short-side<16 at baseline800.',
                        'Target-aware crops require GT and are available during training only; not a deployable inference result or validation gain.',
                        'Maximum IoU geometry is optimistic, not one-to-one recall/AP; frozen model, no gradients.'], 'summary': {}}
    for key in ['best_any_iou', 'best_argmax_class_iou']:
        whole = torch.tensor([r['whole'][key] for r in records])
        zoom = torch.tensor([r['zoom'][key] for r in records])
        report['summary'][key] = {'whole_mean': float(whole.mean()), 'zoom_mean': float(zoom.mean()),
            'zoom_better_by_over005': int((zoom > whole + .05).sum()),
            'zoom_worse_by_over005': int((zoom < whole - .05).sum()),
            'whole_iou75': int((whole >= .75).sum()), 'zoom_iou75': int((zoom >= .75).sum()),
            'whole_iou90': int((whole >= .9).sum()), 'zoom_iou90': int((zoom >= .9).sum())}
    out = ROOT / 'experiments/query_pipeline_audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'zoom_teacher_report.json').write_text(json.dumps(report, indent=2))
    (out / 'zoom_teacher_records.json').write_text(json.dumps(records))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
