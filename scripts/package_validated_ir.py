"""Validate and package the same paired single-detector inference path."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
import numpy as np
from PIL import Image
import torch
import next_ir_reliability
from ir_query_alignment import valid_ir_mask
from src.core import YAMLConfig
from evaluate_variants import encode_txt, decode_txt, evaluate
from faster_coco_eval import COCO


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu_index)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gpu_lock = (ROOT / f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_lock = (out / 'package.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'COMPLETE').exists()
    checkpoint = Path(args.checkpoint)
    assert checkpoint.parent.name == 'ir_reliability_control800' and checkpoint.name == 'weights_epoch_006.pth'
    train_path = ROOT / 'data/annotations/train1600.json'
    val_path = ROOT / 'data/annotations/val400.json'
    train = json.loads(train_path.read_text())
    val = json.loads(val_path.read_text())
    assert len(train['images']) == 1600 and len(val['images']) == 400
    assert not ({i['id'] for i in train['images']} & {i['id'] for i in val['images']})
    cfg_path = ROOT / 'configs/ir_reliability_control800.yml'
    cfg = YAMLConfig(str(cfg_path))
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    assert torch.cuda.mem_get_info()[0] >= 5120 * 1024**2
    model = cfg.model
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = state['ema']['module'] if 'ema' in state else state['model']
    assert 'ema' in state or state.get('source') == 'EMA' or state.get('checkpoint_format') == 'ema_inference', 'Require verified EMA inference weights'
    model.load_state_dict(weights, strict=True)
    del weights
    del state
    assert all(not s.gate_enabled for s in model.ir_samplers)
    model.cuda().eval()
    post = cfg.postprocessor
    post.num_top_queries = 100
    identity = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
        'config': str(cfg_path), 'training_images': 1600, 'selected_epoch': 6,
        'train_manifest_sha256': digest(train_path), 'val_manifest_sha256': digest(val_path),
        'selection': 'Existing independent val400 best checkpoint; no phase2 labels or training',
        'modalities': ['RGB', 'IR'], 'reliability_gate': False, 'size': 800,
        'method': 'single-model native top100; no TTA, NMS or detector ensemble',
        'leaderboard_score': None, 'expected_validation_map': 55.41615840012157,
        'limitations': '1600 training images; local AP does not establish phase2 improvement or 57+',
    }
    atomic(out / 'submission_manifest.json', identity)
    began = time.monotonic()

    def infer(images, image_root, archive_path, stage):
        predictions = {}
        with zipfile.ZipFile(archive_path) as archive:
            names = [n for n in archive.namelist() if Path(n).parent.name == 'infrared' and not n.endswith('/')]
            by_name = {Path(n).name: n for n in names}
            assert len(by_name) == len(names)
            assert {Path(i['file_name']).name for i in images} <= set(by_name)
            with torch.inference_mode():
                for index, info in enumerate(images):
                    with Image.open(image_root / info['file_name']) as image:
                        rgb = image.convert('RGB')
                    with Image.open(io.BytesIO(archive.read(by_name[Path(info['file_name']).name]))) as image:
                        ir = image.convert('RGB')
                    w, h = rgb.size
                    assert (w, h) == (info['width'], info['height']) == ir.size
                    def tensor(image):
                        a = np.asarray(image.resize((800, 800), Image.Resampling.BILINEAR)).copy()
                        return torch.from_numpy(a).permute(2, 0, 1).float() / 255
                    x = torch.cat([tensor(rgb), tensor(ir), valid_ir_mask(ir, 800)])[None].cuda()
                    with torch.autocast('cuda', dtype=torch.float16):
                        result = post(model(x), torch.tensor([[w, h]], device='cuda'))[0]
                    boxes = result['boxes'].float().cpu()
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    pred = {'boxes': boxes[keep].tolist(), 'scores': result['scores'].float().cpu()[keep].tolist(),
                            'labels': result['labels'].cpu()[keep].tolist()}
                    txt = encode_txt(pred, w, h)
                    restored = decode_txt(txt, w, h)
                    assert len(restored['labels']) == len(pred['labels'])
                    if pred['boxes']:
                        assert np.max(np.abs(np.asarray(restored['boxes']) - np.asarray(pred['boxes']))) < 1e-4
                    predictions[str(info['id'])] = restored
                    if (index + 1) % 25 == 0:
                        progress = {'stage': stage, 'images': index + 1, 'total': len(images),
                                    'seconds': time.monotonic() - began, 'time': time.time()}
                        atomic(out / 'status.json', progress)
                        print(json.dumps(progress), flush=True)
        return predictions

    train_archive = Path(cfg.yaml_cfg['val_dataloader']['dataset']['archive'])
    val_predictions = infer(val['images'], ROOT / 'data/train', train_archive, 'validation')
    result = evaluate(COCO(str(val_path)), val_predictions)
    identity['packaging_path_validation'] = result
    atomic(out / 'validation.json', result)
    (out / 'validation_predictions.json').write_text(json.dumps(val_predictions))
    assert abs(result['map'] - identity['expected_validation_map']) < .1, 'Packaging path failed to reproduce validation AP'
    atomic(out / 'submission_manifest.json', identity)
    phase_path = ROOT / 'data/phase2/images.json'
    images = json.loads(phase_path.read_text())['images']
    assert len(images) == 1000
    predictions = infer(images, ROOT / 'data/phase2', ROOT / 'AIC2026_PHASE_2_1000.zip', 'phase2_inference')
    folder = out / 'prediction_txt'
    folder.mkdir(exist_ok=True)
    package = out / 'submission.zip'
    with zipfile.ZipFile(package.with_suffix('.tmp'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        seen = set()
        for info in images:
            name = Path(info['file_name']).stem + '.txt'
            assert name not in seen
            seen.add(name)
            txt = encode_txt(predictions[str(info['id'])], info['width'], info['height'])
            decode_txt(txt, info['width'], info['height'])
            (folder / name).write_text(txt)
            archive.writestr(name, txt)
    package.with_suffix('.tmp').replace(package)
    with zipfile.ZipFile(package) as archive:
        assert len(archive.namelist()) == 1000 and archive.testzip() is None
        for info in images:
            decode_txt(archive.read(Path(info['file_name']).stem + '.txt').decode(), info['width'], info['height'])
    identity.update(images=1000, images_manifest_sha256=digest(phase_path), submission_sha256=digest(package),
                    elapsed_seconds=time.monotonic() - began, peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2, memory_cap_gib=4,
                    validation='Same inference path reproduced val400; strict EMA load; 1000 valid paired TXT; <=100 boxes/image; coordinate roundtrip and ZIP CRC passed')
    (out / 'raw_predictions.json').write_text(json.dumps(predictions))
    atomic(out / 'submission_manifest.json', identity)
    atomic(out / 'status.json', {'stage': 'complete', 'images': 1000, 'time': time.time()})
    (out / 'COMPLETE').write_text('Ready for user submission; not submitted or scored\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
