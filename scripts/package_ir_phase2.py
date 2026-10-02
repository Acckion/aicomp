"""Official phase2 paired inference only; one detector, native top100, no TTA."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'D-FINE'))
import numpy as np
from PIL import Image
import torch
import ir_content_alignment
from ir_query_alignment import valid_ir_mask
from src.core import YAMLConfig
from evaluate_variants import encode_txt, decode_txt


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-index', type=int, default=3)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu_index), 'GPU visibility must match the locked physical card'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_lock = (out / 'package.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'COMPLETE').exists(), 'Package already complete'
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024**3 / total)
    free, _ = torch.cuda.mem_get_info()
    assert free >= 9216 * 1024**2, 'Insufficient free memory'
    checkpoint = Path(args.checkpoint)
    match = re.fullmatch(r'weights_epoch_(\d+)\.pth', checkpoint.name)
    assert match and checkpoint.parent.name == 'full2000_ir_content800'
    selected_epoch = int(match.group(1))
    image_manifest = ROOT / 'data/phase2/images.json'
    images = json.loads(image_manifest.read_text())['images']
    assert len(images) == 1000
    identity = {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
        'images_manifest_sha256': hashlib.sha256(image_manifest.read_bytes()).hexdigest(),
        'size': 800, 'modalities': ['RGB', 'IR'],
        'method': 'single-model native top100; no TTA or NMS',
        'training_images': 2000, 'selected_epoch': selected_epoch,
        'selection': 'Predeclared in experiments/full2000_ir/selection_policy.json',
        'leaderboard_score': None,
    }
    atomic(out / 'submission_manifest.json', identity)
    cfg = YAMLConfig(str(ROOT / 'configs/full2000_ir_content800.yml'))
    model = cfg.model
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = state['ema']['module'] if 'ema' in state else state['model']
    model.load_state_dict(weights, strict=True)
    del weights, state
    model.cuda().eval()
    post = cfg.postprocessor
    post.num_top_queries = 100
    folder = out / 'prediction_txt'
    folder.mkdir(exist_ok=True)
    began = time.monotonic()
    predictions = {}
    with zipfile.ZipFile(ROOT / 'AIC2026_PHASE_2_1000.zip') as source:
        members = [n for n in source.namelist() if Path(n).parent.name == 'infrared' and not n.endswith('/')]
        assert len(members) == 1000
        by_name = {Path(n).name: n for n in members}
        assert len(by_name) == 1000
        assert {Path(i['file_name']).name for i in images} == set(by_name)
        identity['ir_archive_entries_sha256'] = hashlib.sha256(json.dumps(
            [(n, source.getinfo(n).CRC, source.getinfo(n).file_size) for n in sorted(members)]
        ).encode()).hexdigest()
        with torch.inference_mode():
            for index, info in enumerate(images):
                with Image.open(ROOT / 'data/phase2' / info['file_name']) as image:
                    rgb = image.convert('RGB')
                with Image.open(io.BytesIO(source.read(by_name[Path(info['file_name']).name]))) as image:
                    ir = image.convert('RGB')
                w, h = rgb.size
                assert (w, h) == (info['width'], info['height']) == ir.size
                def tensor(image):
                    array = np.asarray(image.resize((800, 800), Image.Resampling.BILINEAR)).copy()
                    return torch.from_numpy(array).permute(2, 0, 1).float() / 255
                x = torch.cat([tensor(rgb), tensor(ir), valid_ir_mask(ir, 800)])[None].cuda()
                # Match the existing AMP detector validation path and paired
                # dataset interpolation; all geometry stays in RGB coordinates.
                with torch.autocast('cuda', dtype=torch.float16):
                    result = post(model(x), torch.tensor([[w, h]], device='cuda'))[0]
                boxes = result['boxes'].float().cpu()
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                pred = {'boxes': boxes[keep].tolist(),
                        'scores': result['scores'].float().cpu()[keep].tolist(),
                        'labels': result['labels'].cpu()[keep].tolist()}
                text = encode_txt(pred, w, h)
                restored = decode_txt(text, w, h)
                assert len(restored['labels']) == len(pred['labels'])
                if pred['boxes']:
                    assert np.max(np.abs(np.asarray(restored['boxes']) - np.asarray(pred['boxes']))) < 1e-4
                name = Path(info['file_name']).stem + '.txt'
                (folder / name).write_text(text)
                predictions[str(info['id'])] = pred
                if (index + 1) % 25 == 0:
                    progress = {'stage': 'inference', 'images': index + 1, 'total': 1000,
                                'seconds': time.monotonic() - began, 'time': time.time()}
                    atomic(out / 'status.json', progress)
                    print(json.dumps(progress), flush=True)
    package = out / 'submission.zip'
    with zipfile.ZipFile(package.with_suffix('.tmp'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for info in images:
            name = Path(info['file_name']).stem + '.txt'
            archive.write(folder / name, name)
    package.with_suffix('.tmp').replace(package)
    with zipfile.ZipFile(package) as archive:
        assert len(archive.namelist()) == 1000 and archive.testzip() is None
        for info in images:
            decode_txt(archive.read(Path(info['file_name']).stem + '.txt').decode(), info['width'], info['height'])
    identity.update(images=1000, submission_sha256=hashlib.file_digest(package.open('rb'), 'sha256').hexdigest(),
                    inference_seconds=time.monotonic()-began,
                    validation='1000 matching RGB/IR pairs; strict checkpoint load; valid 12 classes, normalized boxes, confidence; <=100/image; TXT roundtrip and ZIP CRC passed')
    atomic(out / 'submission_manifest.json', identity)
    atomic(out / 'status.json', {'stage': 'complete', 'images': 1000, 'time': time.time()})
    (out / 'raw_predictions.json').write_text(json.dumps(predictions))
    (out / 'COMPLETE').write_text('Ready for user submission; not submitted or scored\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
