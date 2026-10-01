"""Train-only fixed-model matching comparison; does not train or alter scores."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import functional as TF
from torchvision.ops import box_iou, box_convert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
from src.core import YAMLConfig


@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4', 'Run with CUDA_VISIBLE_DEVICES=4; the lock protects physical GPU4.'
    output = ROOT / 'experiments/matching_quality_audit'
    output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'experiments/mechanism_trials/gpu4.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / 'COMPLETE').unlink(missing_ok=True)
    torch.set_num_threads(2); torch.manual_seed(20261002)
    torch.cuda.set_per_process_memory_fraction(3 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark = False
    cfg = YAMLConfig(str(ROOT / 'configs/rgb1600.yml'), eval_spatial_size=[800, 800])
    model = cfg.model
    path = ROOT / 'runs/ft_aug800/weights_epoch_020.pth'
    state = torch.load(path, map_location='cpu', weights_only=False)
    weights = state['ema']['module'] if 'ema' in state else state['model']
    weights = {k: v for k, v in weights.items() if k not in ['decoder.anchors', 'decoder.valid_mask']}
    missing, extra = model.load_state_dict(weights, strict=False)
    assert not extra and set(missing) <= {'decoder.anchors', 'decoder.valid_mask'}
    model.cuda().eval(); matcher = cfg.criterion.matcher.cuda()
    annotation = ROOT / 'data/annotations/train1600.json'
    data = json.loads(annotation.read_text())
    by_image = {m['id']: [] for m in data['images']}
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    # Fixed evenly spaced sample, not selected by observed errors.
    indices = torch.linspace(0, len(data['images'])-1, 128).round().long().tolist()
    records = []
    for index in indices:
        meta = data['images'][index]
        annotations = [a for a in by_image[meta['id']] if not a.get('iscrowd', 0)]
        if not annotations:
            continue
        image = Image.open(ROOT / 'data/train' / meta['file_name']).convert('RGB')
        assert image.size == (meta['width'], meta['height'])
        sample = TF.to_tensor(TF.resize(image, [800, 800])).unsqueeze(0).cuda()
        prediction = model(sample)
        boxes = torch.tensor([a['bbox'] for a in annotations], dtype=torch.float32, device='cuda')
        # Match CocoDetection training target policy, including border boxes.
        boxes = box_convert(boxes, 'xywh', 'xyxy')
        boxes[:, 0::2].clamp_(0, meta['width'])
        boxes[:, 1::2].clamp_(0, meta['height'])
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        annotations = [a for a, valid in zip(annotations, keep.cpu().tolist()) if valid]
        if not annotations:
            continue
        boxes = box_convert(boxes[keep], 'xyxy', 'cxcywh') / boxes.new_tensor([meta['width'], meta['height']]*2)
        labels = torch.tensor([a['category_id'] for a in annotations], device='cuda')
        target = {'boxes': boxes, 'labels': labels}
        native_q, native_g = matcher(prediction, [target])['indices'][0]
        iou = box_iou(box_convert(prediction['pred_boxes'][0], 'cxcywh', 'xyxy'),
                      box_convert(boxes, 'cxcywh', 'xyxy'))
        probabilities = prediction['pred_logits'][0].sigmoid()
        # Rank-DETR high-order matching objective. This is a diagnostic only,
        # not a full Rank-DETR implementation or an improvement in model AP.
        quality = probabilities[:, labels] * iou.pow(4)
        alt_q, alt_g = linear_sum_assignment(-quality.cpu().numpy())
        original = dict(zip(native_g.tolist(), native_q.tolist()))
        alternate = dict(zip(alt_g.tolist(), alt_q.tolist()))
        for g, annotation_row in enumerate(annotations):
            q, aq = original[g], alternate[g]
            records.append({'image_id': meta['id'], 'annotation_id': annotation_row['id'],
                            'class': int(labels[g]), 'short_side800': float(boxes[g, 2:].min()*800),
                            'native_iou': float(iou[q, g]), 'high_order_iou': float(iou[aq, g]),
                            'changed_query': q != aq,
                            'native_argmax_correct': int(probabilities[q].argmax()) == int(labels[g]),
                            'high_order_argmax_correct': int(probabilities[aq].argmax()) == int(labels[g]),
                            'native_score': float(probabilities[q, labels[g]]),
                            'high_order_score': float(probabilities[aq, labels[g]])})
        if len(records) % 10 == 0:
            print(json.dumps({'images_index': index, 'targets': len(records)}), flush=True)

    def summary(rows):
        n = len(rows)
        if not n:
            return {'targets': 0}
        return {'targets': n, 'changed_queries': sum(r['changed_query'] for r in rows),
                'mean_iou_native': sum(r['native_iou'] for r in rows)/n,
                'mean_iou_high_order': sum(r['high_order_iou'] for r in rows)/n,
                'improved_over005': sum(r['high_order_iou']-r['native_iou']>.05 for r in rows),
                'worsened_over005': sum(r['native_iou']-r['high_order_iou']>.05 for r in rows),
                'argmax_correct_native': sum(r['native_argmax_correct'] for r in rows)/n,
                'argmax_correct_high_order': sum(r['high_order_argmax_correct'] for r in rows)/n,
                'thresholds': {str(t): {'native': sum(r['native_iou']>=t for r in rows),
                                      'high_order': sum(r['high_order_iou']>=t for r in rows),
                                      'gained': sum(r['native_iou']<t<=r['high_order_iou'] for r in rows),
                                      'lost': sum(r['high_order_iou']<t<=r['native_iou'] for r in rows)}
                               for t in [.5, .75, .9]}}
    report = {'training_images_sampled': len(indices), 'all': summary(records),
              'short_side_under16': summary([r for r in records if r['short_side800']<16]),
              'per_class': {c['name']: summary([r for r in records if r['class']==c['id']])
                            for c in data['categories']},
              'checkpoint_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
              'annotations_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
              'notes': ['Evaluation-mode train1600 views, fixed parent model, no augmentations or gradients.',
                        'Both assignments are one-to-one. GT is used for training-only mechanism analysis.',
                        'Higher assigned IoU is not a measured AP improvement. Encoder/GO/DN changes need separate tests.',
                        'No test input, corrected predictions or submission packages.']}
    (output / 'records.json').write_text(json.dumps(records))
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'COMPLETE').write_text('ok\n')
    print(json.dumps(report['all']), flush=True)


if __name__ == '__main__':
    main()
