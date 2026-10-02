"""Train-only source vocabulary support; no crops, fitting or heldout images."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'D-FINE'), str(ROOT / 'scripts')]
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert, box_iou
from src.core import YAMLConfig

# Explicit raw category IDs from OpenMMLab's original Objects365 v2 name table.
# Unknown/general categories are not assigned a narrow source category.
EXPECTED = {0: 1, 1: 22, 3: 3, 4: 90, 5: 47, 6: 6, 7: 157, 9: 45, 11: 184}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    lock = (ROOT / f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT / 'experiments/obj365_class_rows'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    cfg = YAMLConfig(str(ROOT / 'configs/scene_rgb800.yml'), num_classes=366)
    model = cfg.model
    state = torch.load(ROOT / 'checkpoints/dfine_x_obj365.pth', map_location='cpu', weights_only=False)
    weights = dict(state['ema']['module'] if 'ema' in state else state['model'])
    for key in ('decoder.anchors', 'decoder.valid_mask'):
        if key in model.state_dict():
            weights[key] = model.state_dict()[key]
    model.load_state_dict(weights, strict=True)
    assert model.decoder.enc_score_head.out_features == 366
    model.cuda().eval()
    data = json.loads((ROOT / 'data/annotations/scene_train.json').read_text())
    heldout = json.loads((ROOT / 'data/annotations/scene_val.json').read_text())
    train_ids = {i['id'] for i in data['images']}
    assert train_ids.isdisjoint(i['id'] for i in heldout['images'])
    annotations = defaultdict(list)
    by_class = defaultdict(set)
    for a in data['annotations']:
        annotations[a['image_id']].append(a)
        by_class[a['category_id']].add(a['image_id'])
    rng = random.Random(20260929)
    selected = set()
    for category in sorted(by_class):
        ids = sorted(by_class[category]); rng.shuffle(ids)
        selected.update(ids[:10])
    remaining = sorted(train_ids - selected); rng.shuffle(remaining)
    selected.update(remaining[:192 - len(selected)])
    records = []
    start = time.time()
    with torch.inference_mode():
        for image in sorted(data['images'], key=lambda i: i['id']):
            if image['id'] not in selected or not annotations[image['id']]:
                continue
            with Image.open(ROOT / 'data/train' / image['file_name']) as im:
                x = TF.to_tensor(TF.resize(im.convert('RGB'), [800, 800])).unsqueeze(0).cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                prediction = model(x)
            boxes = box_convert(prediction['pred_boxes'][0].float(), 'cxcywh', 'xyxy')
            probabilities = prediction['pred_logits'][0].float().sigmoid()
            assert torch.isfinite(boxes).all() and torch.isfinite(probabilities).all()
            for a in annotations[image['id']]:
                bx, by, bw, bh = a['bbox']
                gt = boxes.new_tensor([[bx, by, bx+bw, by+bh]]) / boxes.new_tensor([image['width'], image['height']]*2)
                ious = box_iou(boxes, gt).flatten()
                best = int(ious.argmax())
                source = EXPECTED.get(a['category_id'])
                correct = probabilities.argmax(-1) == source if source is not None else None
                records.append({'image_id':image['id'], 'annotation_id':a['id'], 'target_class':a['category_id'],
                    'best_any_iou':float(ious[best]), 'best_geometry_source_row':int(probabilities[best].argmax()),
                    'expected_row':source,
                    'expected_probability_at_best_geometry':float(probabilities[best, source]) if source is not None else None,
                    'expected_argmax_best_iou':float(ious[correct].max()) if correct is not None and correct.any() else 0.0})
            if len(selected) and len(records) % 20 == 0:
                print('AUDIT_PROGRESS', image['id'], len(records), flush=True)
    summary = {}
    for category in sorted(by_class):
        rows = [r for r in records if r['target_class']==category]
        supported = [r for r in rows if r['best_any_iou']>=.5]
        summary[category] = {'targets':len(rows), 'geometry_iou50':len(supported),
            'expected_argmax_iou50':sum(r['expected_argmax_best_iou']>=.5 for r in rows),
            'top_source_rows_at_supported_geometry':Counter(r['best_geometry_source_row'] for r in supported).most_common(5),
            'expected_probability_mean':sum(r['expected_probability_at_best_geometry'] or 0 for r in supported)/max(len(supported),1)}
    report = {'stage':'complete','images':len(selected),'seconds':time.time()-start,'expected_rows':EXPECTED,
        'source':'public Objects365-only X, no fine-tuning','peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'summary':summary,'limitations':'Stratified train-only optimistic geometry and source-row support, not AP or unseen validation. Chair/Traffic Sign/Other Balls are partial priors. Animal/light/UAV deliberately unmapped.'}
    (out / 'records.json').write_text(json.dumps(records))
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
