"""Fixed-weight validation of historical native inputs and square resizing.

No training or test inputs. Padding is undone before original-coordinate COCO
evaluation. Dynamic anchors/positions are checked against fixed-size inference.
"""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
from src.core import YAMLConfig
from evaluate_variants import select, coco_rows
from faster_coco_eval import COCO, COCOeval_faster

OUT = ROOT / 'experiments/input_protocol_audit'
ANN = ROOT / 'data/annotations/val400.json'
WEIGHT = ROOT / 'runs/ft_aug800/weights_epoch_020.pth'


def atom(path, value):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def dimensions(width, height, mode):
    if mode == 'square800':
        rw, rh = 800, 800
    elif mode == 'aspect_equal_area800':
        scale = 800 / math.sqrt(width * height)
        rw, rh = round(width * scale), round(height * scale)
    elif mode == 'native_pad32':
        rw, rh = width, height
    else:
        raise ValueError(mode)
    return rw, rh, math.ceil(rw / 32) * 32, math.ceil(rh / 32) * 32


def restore(boxes, width, height, shape):
    rw, rh, pw, ph = shape
    # Postprocessor has already converted normalized boxes to padded pixels.
    boxes = boxes.clone()
    boxes[:, [0, 2]] /= rw / width
    boxes[:, [1, 3]] /= rh / height
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes


def evaluate(gt, predictions):
    ev = COCOeval_faster(gt, gt.loadRes(coco_rows(predictions)), 'bbox')
    ev.params.imgIds = sorted(map(int, predictions))
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    precision = ev.eval['precision']
    ap90 = precision[8, :, :, 0, 2]
    per_class = {}
    for k, category in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, 2]
        per_class[gt.cats[category]['name']] = float(p[p >= 0].mean() * 100)
    return {'map': float(ev.stats[0] * 100), 'ap75': float(ev.stats[2] * 100),
            'ap90': float(ap90[ap90 >= 0].mean() * 100),
            'aps': float(ev.stats[3] * 100), 'per_class': per_class}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4', 'Run with CUDA_VISIBLE_DEVICES=4; the lock protects physical GPU4.'
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'experiments/mechanism_trials/gpu4.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / 'COMPLETE').unlink(missing_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20261002)
    identity = {'checkpoint_sha256': hashlib.file_digest(WEIGHT.open('rb'), 'sha256').hexdigest(),
                'annotations_sha256': hashlib.sha256(ANN.read_bytes()).hexdigest(),
                'validation_images': 400, 'gpu': 4, 'memory_cap_gib': 6,
                'precision': 'FP32; no autocast', 'normalization': 'RGB 0..1',
                'single_model': True, 'test_used': False}
    cache_folder = ROOT / 'experiments/next_stage/infer800_tile0'
    cached_identity = json.loads((cache_folder / 'cache_identity.json').read_text())
    for key in ['checkpoint_sha256', 'annotations_sha256']:
        assert identity[key] == cached_identity[key], key
    atom(OUT / 'identity.json', identity)
    cfg = YAMLConfig(str(ROOT / 'configs/rgb1600.yml'), eval_spatial_size=[800, 800])
    model = cfg.model
    checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    weights = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    weights = {k: v for k, v in weights.items() if k not in ['decoder.anchors', 'decoder.valid_mask']}
    missing, extra = model.load_state_dict(weights, strict=False)
    assert not extra and set(missing) <= {'decoder.anchors', 'decoder.valid_mask'}
    del checkpoint, weights
    model.cuda().eval(); post = cfg.postprocessor; post.num_top_queries = 300
    data = json.loads(ANN.read_text()); images = data['images']
    assert len(images) == 400
    gt = COCO(str(ANN))
    for mode in ['square800', 'aspect_equal_area800', 'native_pad32']:
        # Exact reversible coordinate check, including 1080->1088 padding.
        for width, height in [(1920, 1080), (640, 360)]:
            shape = dimensions(width, height, mode)
            original = torch.tensor([[width*.1, height*.2, width*.8, height*.9]])
            resized = original * original.new_tensor([shape[0]/width, shape[1]/height]*2)
            assert torch.allclose(restore(resized, width, height, shape), original, atol=2e-4)
    # Switch both caches, not merely the decoder anchor size.
    image = Image.open(ROOT / 'data/train' / images[0]['file_name']).convert('RGB')
    sample = TF.to_tensor(TF.resize(image, [800, 800])).unsqueeze(0).cuda()
    with torch.inference_mode():
        fixed = model(sample)
        model.decoder.eval_spatial_size = None
        model.encoder.eval_spatial_size = None
        dynamic = model(sample)
    difference = {k: float((fixed[k] - dynamic[k]).abs().max())
                  for k in ['pred_logits', 'pred_boxes']}
    assert difference['pred_boxes'] < 1e-5 and difference['pred_logits'] < 1e-4, difference
    atom(OUT / 'preflight.json', {'dynamic_fixed_difference': difference, 'coordinate_roundtrip': 'passed'})
    del fixed, dynamic, sample
    results = {}
    cached = json.loads((cache_folder / 'raw_predictions.json').read_text())
    assert set(map(str, gt.getImgIds())) == set(cached)
    results['square800'] = evaluate(gt, {k: select(v) for k, v in cached.items()})
    atom(OUT / 'report.json', {'identity': identity, 'results': results})
    for mode in ['aspect_equal_area800', 'native_pad32']:
        predictions = {}; started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for index, meta in enumerate(images):
                image = Image.open(ROOT / 'data/train' / meta['file_name']).convert('RGB')
                width, height = image.size
                assert (width, height) == (meta['width'], meta['height'])
                shape = dimensions(width, height, mode); rw, rh, pw, ph = shape
                tensor = TF.to_tensor(TF.resize(image, [rh, rw]))
                tensor = F.pad(tensor, (0, pw-rw, 0, ph-rh)).unsqueeze(0).cuda()
                prediction = post(model(tensor), torch.tensor([[pw, ph]], device='cuda'))[0]
                boxes = restore(prediction['boxes'], width, height, shape)
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                predictions[str(meta['id'])] = select({
                    'boxes': boxes[valid].cpu().tolist(),
                    'labels': prediction['labels'][valid].cpu().tolist(),
                    'scores': prediction['scores'][valid].cpu().tolist()})
                if (index + 1) % 20 == 0:
                    state = {'stage': 'inference', 'mode': mode, 'images': index+1, 'total': 400,
                             'seconds': time.monotonic()-started, 'pid': os.getpid(),
                             'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
                    atom(OUT / 'status.json', state); print(json.dumps(state), flush=True)
                del tensor, prediction, boxes
        atom(OUT / f'{mode}_predictions.json', predictions)
        results[mode] = {**evaluate(gt, predictions), 'seconds': time.monotonic()-started,
                         'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
        atom(OUT / 'report.json', {'identity': identity, 'results': results,
             'note': 'Fixed checkpoint and val400 only. Native input is a historical-protocol audit; no leaderboard gain claimed.'})
    atom(OUT / 'status.json', {'stage': 'complete', 'pid': os.getpid(), 'time': time.time()})
    (OUT / 'COMPLETE').write_text('ok\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        atom(OUT / 'status.json', {'stage': 'failed', 'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
