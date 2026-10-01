"""Read official modalities from ZIPs without extracting or altering any images.

Headers of all training/phase2 images establish actual encoding and pairing.
Pixel statistics use a seeded val400 sample only, with known RGB predictions
for diagnostic groups. These are signal/coverage proxies, not detector gains,
registration estimates, or calibrated temperatures/depths for JPEG imagery.
"""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / '初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip'
OUT = ROOT / 'experiments/modal_audit'


def header_audit(path):
    modes = defaultdict(Counter)
    dimensions = defaultdict(Counter)
    members = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if item.is_dir() or len(parts) < 2 or parts[-2] not in ('visible', 'infrared', 'depth'):
                continue
            modality = parts[-2]
            key = Path(item.filename).name
            if (modality, key) in members:
                raise ValueError('Duplicate modality filename')
            members[modality, key] = item.filename
            with archive.open(item) as stream, Image.open(stream) as image:
                modes[modality][f'{Path(key).suffix.lower()}:{image.mode}'] += 1
                dimensions[modality][str(image.size)] += 1
        names = {m: {key for modality, key in members if modality == m}
                 for m in ['visible', 'infrared', 'depth']}
        assert names['visible'] == names['infrared'] == names['depth'], 'Missing modality pairs'
    return {'archive': str(path), 'paired_images': len(names['visible']),
            'header_modes': {k: dict(v) for k, v in modes.items()},
            'dimensions': {k: dict(v) for k, v in dimensions.items()},
            'note': 'Matching image dimensions do not establish geometric registration.'}, members


def median(values):
    return float(np.median(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=128)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train_headers, members = header_audit(TRAIN)
    phase2_headers, _ = header_audit(ROOT / 'AIC2026_PHASE_2_1000.zip')
    print(json.dumps({'train_headers': train_headers, 'phase2_headers': phase2_headers}), flush=True)
    data = json.loads((ROOT / 'data/annotations/val400.json').read_text())
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        annotations[annotation['image_id']].append(annotation)
    images = sorted(data['images'], key=lambda i: i['id'])
    selected = random.Random(20261001).sample(images, min(args.sample, len(images)))
    cache = ROOT / 'experiments/mechanism_followup/bn_adapt800/raw_predictions.json'
    raw = json.loads(cache.read_text()) if cache.exists() else None
    if raw is not None:
        import torch
        from torchvision.ops import box_iou
        torch.set_num_threads(2)
    image_rows, object_rows = [], []
    with zipfile.ZipFile(TRAIN) as archive:
        for i, metadata in enumerate(selected):
            name = Path(metadata['file_name']).name
            arrays = {}
            for modality in ['visible', 'infrared', 'depth']:
                with Image.open(io.BytesIO(archive.read(members[modality, name]))) as image:
                    arrays[modality] = np.array(image)
            rgb, ir, depth = [arrays[k] for k in ['visible', 'infrared', 'depth']]
            assert rgb.shape[:2] == ir.shape[:2] == depth.shape[:2], name
            gray = ir.astype(np.float32).mean(-1) if ir.ndim == 3 else ir.astype(np.float32)
            channel_gap = np.abs(ir[..., 0].astype(float) - ir[..., 1].astype(float)).mean() if ir.ndim == 3 else 0.0
            # Only truly single-channel >8-bit imagery has documented mm units.
            metric_depth = depth.ndim == 2 and depth.dtype.itemsize >= 2
            valid = (depth >= 300) & (depth < 19999) if metric_depth else None
            image_rows.append({'id': metadata['id'], 'filename': name,
                               'rgb_mean': float(rgb.mean()), 'ir_channel_mean_difference': float(channel_gap),
                               'ir_zero_fraction': float((gray == 0).mean()), 'ir_std': float(gray.std()),
                               'depth_dtype': str(depth.dtype), 'depth_channels': 1 if depth.ndim == 2 else depth.shape[-1],
                               'metric_depth': metric_depth,
                               'depth_usable_fraction': float(valid.mean()) if metric_depth else None,
                               'depth_cap_fraction': float((depth == 19999).mean()) if metric_depth else None})
            height, width = gray.shape
            gt = annotations[metadata['id']]
            covered = None
            if raw is not None and gt:
                prediction = raw[str(metadata['id'])]
                order = np.argsort(prediction['scores'])[::-1][:100]
                boxes = torch.tensor([prediction['boxes'][j] for j in order], dtype=torch.float32).reshape(-1, 4)
                ground = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                        a['bbox'][1] + a['bbox'][3]] for a in gt], dtype=torch.float32)
                ious = box_iou(boxes, ground).numpy()
                labels = np.array([prediction['labels'][j] for j in order])
                covered = [(float(ious[labels == a['category_id'], j].max())
                            if (labels == a['category_id']).any() else 0.0) for j, a in enumerate(gt)]
            for j, annotation in enumerate(gt):
                x, y, w, h = annotation['bbox']
                x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
                x1, y1 = min(width, int(np.ceil(x+w))), min(height, int(np.ceil(y+h)))
                if x1 <= x0 or y1 <= y0:
                    continue
                pad = max(2, round(min(w, h) * .25))
                left, top, right, bottom = max(0, x0-pad), max(0, y0-pad), min(width, x1+pad), min(height, y1+pad)
                region = gray[top:bottom, left:right]
                ring = np.ones(region.shape, dtype=bool)
                ring[y0-top:y1-top, x0-left:x1-left] = False
                foreground = gray[y0:y1, x0:x1]
                background = region[ring]
                fg_median = float(np.median(foreground))
                bg_median = float(np.median(background)) if background.size else None
                contrast = abs(fg_median-bg_median) if bg_median is not None else None
                bg_mad = float(np.median(np.abs(background-bg_median))) if bg_median is not None else None
                object_rows.append({'image_id': metadata['id'], 'category_id': annotation['category_id'],
                                    'short_side800': min(w*800/width, h*800/height),
                                    'rgb_top100_best_same_class_iou': covered[j] if covered is not None else None,
                                    'ir_box_ring_median_difference': contrast,
                                    'ir_box_ring_contrast_over_mad': contrast/max(bg_mad, 1) if contrast is not None else None,
                                    'depth_usable_fraction': float(valid[y0:y1, x0:x1].mean()) if metric_depth else None})
            if (i+1) % 32 == 0:
                print(f'Decoded {i+1}/{len(selected)} triplets', flush=True)
    groups = {'all': object_rows, 'short_side800_under16': [o for o in object_rows if o['short_side800'] < 16],
              'rgb_no_same_class_iou75': [o for o in object_rows if o['rgb_top100_best_same_class_iou'] is not None and o['rgb_top100_best_same_class_iou'] < .75]}
    summary = {}
    for label, group in groups.items():
        depth_values = [o['depth_usable_fraction'] for o in group if o['depth_usable_fraction'] is not None]
        summary[label] = {'objects': len(group), 'metric_depth_objects': len(depth_values),
                          'median_depth_usable_fraction': median(depth_values),
                          'fraction_metric_objects_under10pct_usable_depth': float(np.mean(np.array(depth_values) < .1)) if depth_values else None,
                          'median_ir_box_ring_difference': median([o['ir_box_ring_median_difference'] for o in group if o['ir_box_ring_median_difference'] is not None]),
                          'median_ir_box_ring_contrast_over_mad': median([o['ir_box_ring_contrast_over_mad'] for o in group if o['ir_box_ring_contrast_over_mad'] is not None])}
    report = {'train_headers': train_headers, 'phase2_headers': phase2_headers,
              'sample': {'split': 'val400', 'seed': 20261001, 'images': len(image_rows)},
              'object_groups': summary, 'image_rows': image_rows, 'object_rows': object_rows,
              'notes': ['No image extraction; ZIP-backed inspection leaves datasets and running jobs unchanged.',
                        '8-bit RGB depth imagery is a separate encoding; millimetre units cannot be inferred.',
                        'Depth value 19999 is treated as range-cap/ambiguous, not reliable metric depth.',
                        'IR local contrast and depth coverage are proxies, not proof of semantic complementarity.',
                        'RGB IoU coverage permits repeated matching; it is not one-to-one recall or AP.',
                        'No phase2 labels or training. Test headers only; all pixel/GT diagnostics use val400.']}
    path = OUT / 'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'report': str(path), 'object_groups': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
