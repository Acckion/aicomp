"""Trace proposal/query/top100 geometry on val only; never correct submissions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'D-FINE'), str(ROOT / 'scripts')]
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert, box_iou
from src.core import YAMLConfig


def main(limit):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = YAMLConfig(str(ROOT / 'configs/ft_aug800_shared3.yml'))
    model = cfg.model
    state = torch.load(ROOT / 'runs/ft_aug800/weights_epoch_020.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.cuda().eval()
    decoder = model.decoder
    original = decoder._select_topk
    captured = {}

    def select(memory, logits, anchors, topk):
        if decoder.query_select_method != 'default':
            raise RuntimeError('Audit expects original max-class topk selection')
        candidate_boxes = (decoder.enc_bbox_head(memory) + anchors).sigmoid()[0]
        indices = logits[0].max(-1).values.topk(topk).indices
        captured['all'] = candidate_boxes[torch.isfinite(anchors[0]).all(-1)]
        captured['selected'] = candidate_boxes[indices]
        return original(memory, logits, anchors, topk)

    decoder._select_topk = select
    ann = json.loads((ROOT / 'data/annotations/val400.json').read_text())
    by_image = defaultdict(list)
    for a in ann['annotations']:
        by_image[a['image_id']].append(a)
    records = []
    started = time.monotonic()
    images = ann['images'][:limit] if limit else ann['images']
    out = ROOT / 'experiments/query_pipeline_audit'
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, info in enumerate(images):
            image = Image.open(ROOT / 'data/train' / info['file_name']).convert('RGB')
            prediction = model(TF.to_tensor(TF.resize(image, [800, 800])).unsqueeze(0).cuda())
            final = box_convert(prediction['pred_boxes'][0].float(), 'cxcywh', 'xyxy').clamp(0, 1)
            probabilities = prediction['pred_logits'][0].sigmoid()
            classes = probabilities.argmax(-1)
            scores, pair_indices = probabilities.flatten().topk(100)
            query_indices, pair_classes = pair_indices // 12, pair_indices % 12
            targets = by_image[info['id']]
            if not targets:
                continue
            gt = torch.tensor([a['bbox'] for a in targets], device='cuda', dtype=torch.float32)
            gt = box_convert(gt, 'xywh', 'xyxy') / gt.new_tensor([info['width'], info['height']] * 2)
            matrix = box_iou(final, gt)
            best_final, best_indices = matrix.max(0)
            all_iou = box_iou(box_convert(captured['all'].float(), 'cxcywh', 'xyxy').clamp(0, 1), gt).max(0).values
            selected_iou = box_iou(box_convert(captured['selected'].float(), 'cxcywh', 'xyxy').clamp(0, 1), gt).max(0).values
            for k, target in enumerate(targets):
                category = target['category_id']
                same = classes == category
                top_same = pair_classes == category
                same_iou = matrix[same, k].max() if same.any() else matrix.new_tensor(0)
                top_iou = matrix[query_indices[top_same], k].max() if top_same.any() else matrix.new_tensor(0)
                q = int(best_indices[k])
                same_score = probabilities[q, category]
                global_rank = int((probabilities > same_score).sum()) + 1
                gt_wh = gt[k, 2:] - gt[k, :2]
                pred_wh = final[q, 2:] - final[q, :2]
                center_error = ((final[q, :2] + final[q, 2:] - gt[k, :2] - gt[k, 2:]) / 2).abs() / gt_wh.clamp_min(1e-6)
                size_error = (pred_wh - gt_wh).abs() / gt_wh.clamp_min(1e-6)
                records.append({'image_id': info['id'], 'annotation_id': target['id'], 'category_id': category,
                    'short_side_800': float(gt_wh.min() * 800), 'gt_area_original': target['area'],
                    'encoder_all_iou': float(all_iou[k]), 'encoder_selected_iou': float(selected_iou[k]),
                    'final_any_class_iou': float(best_final[k]), 'final_argmax_class_iou': float(same_iou),
                    'final_top100_correct_class_iou': float(top_iou),
                    'best_geometry_true_class_global_score_rank': global_rank,
                    'relative_center_error_xy': center_error.tolist(), 'relative_size_error_wh': size_error.tolist()})
            if (index + 1) % 50 == 0:
                print(json.dumps({'images': index + 1, 'gt_boxes': len(records), 'seconds': time.monotonic() - started}), flush=True)
    stages = ['encoder_all_iou', 'encoder_selected_iou', 'final_any_class_iou',
              'final_argmax_class_iou', 'final_top100_correct_class_iou']

    def summarize(rows):
        return {'gt_boxes': len(rows), 'geometry_coverage': {
            stage: {str(threshold): sum(r[stage] >= threshold for r in rows) / len(rows)
                    for threshold in [.5, .75, .9]} for stage in stages},
            'mean_relative_center_error_xy': torch.tensor([r['relative_center_error_xy'] for r in rows]).mean(0).tolist(),
            'mean_relative_size_error_wh': torch.tensor([r['relative_size_error_wh'] for r in rows]).mean(0).tolist(),
            'final_iou75_but_not_top100_class_iou75': sum(r['final_any_class_iou'] >= .75 and r['final_top100_correct_class_iou'] < .75 for r in rows),
            'encoder_iou50_lost_at_topk': sum(r['encoder_all_iou'] >= .5 and r['encoder_selected_iou'] < .5 for r in rows)}

    report = {'images': len(images), 'checkpoint': 'ft_aug800 epoch20 EMA', 'size': 800,
              'seconds': time.monotonic() - started, 'all': summarize(records),
              'short_side_under16': summarize([r for r in records if r['short_side_800'] < 16]),
              'per_class': {c['name']: summarize([r for r in records if r['category_id'] == c['id']])
                            for c in ann['categories'] if any(r['category_id'] == c['id'] for r in records)},
              'notes': ['Geometry coverage is an optimistic per-GT maximum, not one-to-one recall or AP.',
                        'Encoder all-token box head diagnostic includes unselected tokens not directly trained by its original O2O loss.',
                        'Val ground truth only used for diagnosis; no gradients, test labels, score corrections or submission generation.']}
    (out / 'records.json').write_text(json.dumps(records))
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    main(parser.parse_args().limit)
