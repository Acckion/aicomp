"""Local, paired validation-only pilot for the official SenseNova-Vision model.

No fitting, test images, API inference, or detector ensembling. Token likelihood
is an uncalibrated ranking proxy, so geometry/F1 and constant-score AP are also
reported. The small seeded cohort does not estimate official phase2 performance.
"""
import argparse
import fcntl
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/sensenova_probe'
ARCHIVE = ROOT / '初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip'
SOURCE_SHA = '4366e0e1f4d22d2207ecc8e339f886b765ba876e'
REVISION = '79548fcc5b954598799b9317f8d3ec5e347d5c0e'
VARIANTS = ['rgb', 'overlay25', 'paired_ir', 'paired_rgb_control', 'paired_shuffled_ir']
SEED = 20261001


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def parse_boxes(text, tokenizer, trace, categories, width, height):
    # Preserve repeated <p> sections, unlike category-keyed parsers that can
    # overwrite an earlier group. Enforce normalized xyxy; no guessed units.
    ids = [r[0] for r in trace]
    decoded = tokenizer.decode(ids)
    offsets = [0]
    for token in ids:
        offsets.append(offsets[-1] + len(tokenizer.decode([token])))
    alignable = ''.join(tokenizer.decode([t]) for t in ids) == decoded
    result = {'boxes': [], 'scores': [], 'labels': []}
    invalid = []
    names = {c['name'].lower(): c['id'] for c in categories}
    groups = list(re.finditer(r'<p>(.*?)</p>', text, re.S))
    for j, group in enumerate(groups):
        name = ' '.join(group.group(1).lower().strip().split())
        end = groups[j+1].start() if j+1 < len(groups) else len(text)
        for match in re.finditer(r'<bbox>\s*\[([^\]]+)\]\s*</bbox>', text[group.end():end]):
            try:
                values = [float(v.strip()) for v in match.group(1).split(',')]
                assert name in names and len(values) == 4
                x0, y0, x1, y1 = values
                assert all(np.isfinite(values)) and all(0 <= v <= 1 for v in values)
                assert x1 > x0 and y1 > y0
                spans = [(group.start(), group.end()),
                         (group.end()+match.start(), group.end()+match.end())]
                likelihoods = [r[1] for k, r in enumerate(trace)
                               if alignable and any(offsets[k] < right and offsets[k+1] > left
                                                    for left, right in spans)]
                score = float(np.exp(np.mean(likelihoods))) if likelihoods else .5
                result['boxes'].append([x0*width, y0*height, x1*width, y1*height])
                result['labels'].append(names[name])
                result['scores'].append(score)
            except (ValueError, AssertionError):
                invalid.append({'category': name, 'coordinates': match.group(1)})
    order = sorted(range(len(result['scores'])), key=lambda k: -result['scores'][k])[:100]
    return {key: [values[k] for k in order] for key, values in result.items()}, invalid


def summarize(dataset, records):
    from faster_coco_eval import COCO, COCOeval_faster
    from scipy.optimize import linear_sum_assignment
    from torchvision.ops import box_iou
    by_variant = {v: {r['image_id']: r for r in records if r['variant'] == v} for v in VARIANTS}
    common = sorted(set.intersection(*(set(d) for d in by_variant.values())))
    if not common:
        return {'paired_images': 0}
    reference = json.loads((ROOT / 'experiments/mechanism_followup/bn_frozen800/raw_predictions.json').read_text())
    predictions = {v: {i: d[i]['prediction'] for i in common} for v, d in by_variant.items()}
    predictions['dfine_reference'] = {i: reference[str(i)] for i in common}
    gt = COCO()
    gt.dataset = {**dataset, 'images': [i for i in dataset['images'] if i['id'] in common],
                  'annotations': [a for a in dataset['annotations'] if a['image_id'] in common]}
    gt.createIndex()
    result = {}
    for variant, preds in predictions.items():
        metrics = {}
        clipped = {}
        for iid, pred in preds.items():
            order = sorted(range(len(pred['scores'])), key=lambda k: -pred['scores'][k])[:100]
            clipped[iid] = {key: [value[k] for k in order] for key, value in pred.items()}
        for scoring in ['token_likelihood_proxy', 'constant_score']:
            if variant == 'dfine_reference' and scoring == 'constant_score':
                continue
            rows = []
            for iid, pred in clipped.items():
                for b, c, s in zip(pred['boxes'], pred['labels'], pred['scores']):
                    rows.append({'image_id': iid, 'category_id': c,
                                 'bbox': [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                                 'score': float(s) if scoring != 'constant_score' else 1.0})
            if rows:
                ev = COCOeval_faster(gt, gt.loadRes(rows), 'bbox')
                ev.params.imgIds = common
                ev.evaluate(); ev.accumulate(); ev.summarize()
                metrics[scoring if variant != 'dfine_reference' else 'native_scores'] = {
                    'map50_95': float(ev.stats[0]*100), 'ap50': float(ev.stats[1]*100),
                    'ap75': float(ev.stats[2]*100), 'small_ap': float(ev.stats[3]*100)}
            else:
                metrics[scoring] = {'map50_95': 0., 'ap50': 0., 'ap75': 0., 'small_ap': 0.}
        diagnostic = {}
        for threshold in [.5, .75, .9]:
            tp, num_gt, num_pred, recovered, harmed = 0, 0, 0, 0, 0
            for iid, pred in clipped.items():
                annotations = gt.imgToAnns[iid]
                targets = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2],
                                         a['bbox'][1]+a['bbox'][3]] for a in annotations], dtype=torch.float32).reshape(-1, 4)
                labels = np.array([a['category_id'] for a in annotations])
                boxes = torch.tensor(pred['boxes'], dtype=torch.float32).reshape(-1, 4)
                matrix = box_iou(boxes, targets).numpy()
                matrix *= np.array(pred['labels'])[:, None] == labels[None, :]
                matches = np.zeros(len(annotations), dtype=bool)
                if matrix.size:
                    # Maximize the number of threshold-valid one-to-one matches
                    # first, then IoU; this is a geometry diagnostic, not COCO AP.
                    a, b = linear_sum_assignment(-(matrix >= threshold).astype(float) - matrix*.001)
                    matches[b] = matrix[a, b] >= threshold
                baseline = predictions['dfine_reference'][iid]
                keep = np.argsort(baseline['scores'])[::-1][:100]
                ref_boxes = torch.tensor([baseline['boxes'][k] for k in keep], dtype=torch.float32).reshape(-1, 4)
                ref_iou = box_iou(ref_boxes, targets).numpy()
                ref_iou *= np.array([baseline['labels'][k] for k in keep])[:, None] == labels[None, :]
                ref_match = np.zeros(len(annotations), dtype=bool)
                if ref_iou.size:
                    a, b = linear_sum_assignment(-(ref_iou >= threshold).astype(float) - ref_iou*.001)
                    ref_match[b] = ref_iou[a, b] >= threshold
                tp += int(matches.sum()); num_gt += len(annotations); num_pred += len(boxes)
                recovered += int((matches & ~ref_match).sum()); harmed += int((~matches & ref_match).sum())
            diagnostic[str(threshold)] = {'true_positive': tp, 'ground_truth': num_gt, 'predictions': num_pred,
                'precision': tp/max(num_pred, 1), 'recall': tp/max(num_gt, 1),
                'f1': 2*tp/max(num_pred+num_gt, 1), 'gt_covered_only_by_candidate': recovered,
                'gt_covered_only_by_dfine': harmed}
        metrics['one_to_one_geometry'] = diagnostic
        if variant != 'dfine_reference':
            metrics['response_format_errors'] = sum(bool(by_variant[variant][i].get('response_format_error')) for i in common)
        result[variant] = metrics
    summary = {'paired_images': len(common), 'image_ids': common, 'results': result,
               'notes': ['Seeded small val400 pilot, not phase2 score or full validation mAP.',
                         'No confidence calibration/fitting. Token likelihood is a ranking proxy.',
                         'Geometry coverage uses maximum-cardinality same-class matching; not COCO recall.',
                         'D-FINE frozen-BN independent-val reference, same exact images, native detector scores.',
                         'All modalities are zero-shot inputs; no claim of learned RGB/IR fusion.']}
    write_json(OUT / 'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/dev/shm/aicomp_sensenova/model')
    ap.add_argument('--source', default='/dev/shm/aicomp_sensenova/source')
    ap.add_argument('--sample', type=int, default=24)
    ap.add_argument('--max-tokens', type=int, default=3072)
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / 'data/annotations/val400.json'
    dataset = json.loads(dataset_path.read_text())
    images = random.Random(SEED).sample(sorted(dataset['images'], key=lambda i: i['id']), args.sample)
    # Put the smallest sampled frame first for low-memory integration smoke;
    # selection remains the same fixed random cohort, independent of predictions.
    images.sort(key=lambda i: (i['width']*i['height'], i['id']))
    # Official category-task builder wraps each category in <p> tags. Plain
    # category names can route this unified model into segmentation legends.
    category_text = ', '.join(f"<p>{c['name']}</p>" for c in dataset['categories'])
    base_prompt = (f'Detect all instances of {category_text} in the image. Output the results '
                   'as a structured text list with each detection including category and '
                   'bounding box coordinates in <bbox> format.')
    # Keep the official detection request first. A long auxiliary-view preamble
    # caused the unified model to return segmentation palettes instead of boxes.
    paired_prompt = (base_prompt + ' The first image is the detection reference; '
                     'the second image is auxiliary. Return bounding boxes only '
                     'for the first image, in normalized xyxy coordinates.')
    identity = {'model_revision': REVISION, 'source_commit': SOURCE_SHA, 'seed': SEED,
                'annotation_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                'image_ids': [i['id'] for i in images], 'variants': VARIANTS,
                'base_prompt': base_prompt, 'paired_prompt': paired_prompt,
                'dtype': 'bf16', 'overlay_alpha': .25, 'max_tokens': args.max_tokens,
                'score': 'geometric mean generated-token likelihood per detection span; uncalibrated'}
    path = OUT / 'identity.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity, 'Changed pilot protocol; use a fresh experiment folder'
    else:
        write_json(path, identity)
    record_path = OUT / 'records.jsonl'
    records = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
    if args.summarize_only:
        print(json.dumps(summarize(dataset, records), ensure_ascii=False)); return
    if args.plan_only:
        print(json.dumps(identity, ensure_ascii=False)); return
    worker_lock = (OUT / 'worker.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    actual_sha = subprocess.check_output(['git', '-C', args.source, 'rev-parse', 'HEAD'], text=True).strip()
    assert actual_sha == SOURCE_SHA
    assert json.loads((OUT / 'model_identity.json').read_text())['revision'] == REVISION
    sys.path.insert(0, args.source)
    from inference.sensenova_vision import SenseNovaVisionModel
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 4
    for i in range(4):
        torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(i).total_memory, i)

    class SharedMemoryModel(SenseNovaVisionModel):
        def _infer_device_map(self, model):
            # The image-generation expert is not executed in dense_detection.
            # Keep its weights on CPU without changing active BF16 parameters.
            mapping = {'vit_model': 0, 'connector': 0, 'vit_pos_embed': 0,
                       'time_embedder': 0, 'latent_pos_embed': 0, 'vae2llm': 0, 'llm2vae': 0,
                       'language_model.model.embed_tokens': 0, 'language_model.lm_head': 0,
                       'language_model.model.norm': 0, 'language_model.model.norm_moe_gen': 'cpu',
                       'language_model.model.rotary_emb': 0}
            for i, layer in enumerate(model.language_model.model.layers):
                device = 0 if i < 4 else 1 if i < 12 else 2 if i < 20 else 3
                prefix = f'language_model.model.layers.{i}'
                mapping[prefix] = device
                for name, _ in layer.named_modules():
                    if name and 'moe_gen' in name and not any('moe_gen' in p for p in name.split('.')[:-1]):
                        mapping[f'{prefix}.{name}'] = 'cpu'
            print('Detection-only BF16 device map:', mapping, flush=True)
            write_json(OUT / 'device_map.json', mapping)
            return mapping

    write_json(OUT / 'status.json', {'stage': 'loading_model', 'time': time.time()})
    model = SharedMemoryModel(model_path=args.model, device='cuda', dtype='bf16',
                              offload_folder='/dev/shm/aicomp_sensenova/offload')
    trace = []

    def capture(_, inputs, output):
        logits = output[0].float()
        maximum, token = logits.max(dim=-1)
        logp = maximum - torch.logsumexp(logits, dim=-1)
        trace.append((int(token.item()), float(logp.item())))

    hook = model.model.language_model.lm_head.register_forward_hook(capture)
    completed = {(r['image_id'], r['variant']) for r in records}
    with zipfile.ZipFile(ARCHIVE) as archive, record_path.open('a') as stream:
        members = {(Path(p).parts[-2], Path(p).name): p for p in archive.namelist()
                   if len(Path(p).parts) > 1 and Path(p).parts[-2] in ['visible', 'infrared']}

        def load(modality, metadata):
            with Image.open(io.BytesIO(archive.read(members[modality, Path(metadata['file_name']).name]))) as image:
                return image.convert('RGB')

        for index, image_info in enumerate(images):
            rgb, ir = load('visible', image_info), load('infrared', image_info)
            assert rgb.size == ir.size == (image_info['width'], image_info['height'])
            shuffled = load('infrared', images[(index+1) % len(images)]).resize(rgb.size)
            for variant in VARIANTS:
                if (image_info['id'], variant) in completed:
                    continue
                auxiliary = {'paired_ir': ir, 'paired_rgb_control': rgb,
                             'paired_shuffled_ir': shuffled}.get(variant)
                primary = Image.blend(rgb, ir, .25) if variant == 'overlay25' else rgb
                contents = [{'type': 'image', 'value': primary}]
                if auxiliary is not None:
                    contents.append({'type': 'image', 'value': auxiliary})
                prompt = paired_prompt if auxiliary is not None else base_prompt
                contents.append({'type': 'text', 'value': prompt})
                trace.clear()
                for gpu in range(4):
                    torch.cuda.reset_peak_memory_stats(gpu)
                write_json(OUT / 'status.json', {'stage': 'inference', 'image_id': image_info['id'],
                    'variant': variant, 'completed_requests': len(records), 'total_requests': len(images)*len(VARIANTS),
                    'time': time.time()})
                start = time.monotonic()
                with torch.inference_mode():
                    text = model.generate(contents=contents, mode='dense_detection',
                                          noise_seed=SEED+image_info['id'], max_think_token_n=args.max_tokens)
                assert isinstance(text, str)
                format_error = None
                if '<color>' in text and '<bbox>' not in text:
                    format_error = 'segmentation_palette_instead_of_detection'
                elif '<bbox>' not in text and '<p>' in text:
                    format_error = 'category_output_without_boxes'
                # Unsupported auxiliary inputs must not abort the RGB control.
                # Keep raw responses and explicitly count format failures; never
                # reinterpret masks as boxes or silently fall back to RGB.
                # generate_text omits the last predicted token (normally EOS).
                prediction, invalid = parse_boxes(text, model.tokenizer, trace[:-1], dataset['categories'], *rgb.size)
                record = {'image_id': image_info['id'], 'filename': image_info['file_name'],
                          'variant': variant, 'prompt': prompt, 'text': text, 'prediction': prediction,
                          'invalid_detections': invalid, 'response_format_error': format_error,
                          'seconds': time.monotonic()-start,
                          'generated_tokens': len(trace), 'truncated': len(trace) >= args.max_tokens,
                          'peak_reserved_gib': [torch.cuda.max_memory_reserved(i)/1024**3 for i in range(4)],
                          'peak_allocated_gib': [torch.cuda.max_memory_allocated(i)/1024**3 for i in range(4)]}
                stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                records.append(record); completed.add((image_info['id'], variant))
                print(json.dumps({k: record[k] for k in ['image_id', 'variant', 'seconds', 'generated_tokens',
                                                       'peak_reserved_gib', 'response_format_error']}, ensure_ascii=False), flush=True)
                if index < 2:
                    drawing = rgb.copy(); draw = ImageDraw.Draw(drawing)
                    for box, category in zip(prediction['boxes'], prediction['labels']):
                        draw.rectangle(box, outline='orange', width=3)
                        draw.text((box[0], box[1]), dataset['categories'][category]['name'], fill='orange')
                    drawing.thumbnail((1280, 720))
                    drawing.save(OUT / f'{image_info["id"]}_{variant}.jpg', quality=85)
                torch.cuda.empty_cache()
            summary = summarize(dataset, records)
            print(f'Completed paired cohort: {summary["paired_images"]}/{len(images)}', flush=True)
    hook.remove()
    write_json(OUT / 'status.json', {'stage': 'complete', 'completed_requests': len(records),
                                   'paired_images': len(images), 'time': time.time()})
    (OUT / 'COMPLETE').write_text('Validation-only pilot complete\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / 'status.json', {'stage': 'failed', 'error': repr(exc), 'time': time.time()})
        raise
