"""Validate and convert ONLY the official training archive; persist the split."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
import zipfile
import zlib

import numpy as np
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['person', 'boat', 'animal', 'seat', 'sign', 'bicycle', 'car', 'ball',
         'light', 'garbage can', 'uav', 'tricycle']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT / '初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip')
    args = parser.parse_args()
    dest = ROOT / 'data/train'
    annotations = ROOT / 'data/annotations'
    dest.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        members = archive.infolist()
        rgb_members = [m for m in members if any(x.lower() in ('rgb', 'visible', 'color')
                       for x in Path(m.filename).parts[:-1]) and Path(m.filename).suffix.lower() in ('.jpg', '.jpeg', '.png')]
        labels = [m for m in members if Path(m.filename).suffix.lower() == '.txt'
                  and Path(m.filename).stem in {Path(x.filename).stem for x in rgb_members}]
        if len(rgb_members) != 2000 or len(labels) != 2000:
            raise ValueError(f'Expected 2000 RGB and 2000 labels, found {len(rgb_members)}, {len(labels)}. Archive roots: {sorted({Path(m.filename).parts[0] for m in members})}')
        # extract only RGB and labels, leaving IR/depth and ALL test archives untouched.
        for m in rgb_members + labels:
            target = (dest / m.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f'Unsafe archive path: {m.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == m.file_size:
                crc = 0
                with target.open('rb') as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b''):
                        crc = zlib.crc32(chunk, crc)
                if crc == m.CRC:
                    continue
            with archive.open(m) as src, target.open('wb') as out:
                shutil.copyfileobj(src, out)  # ZipExtFile checks CRC at EOF.
    label_paths = {Path(m.filename).stem: dest / m.filename for m in labels}
    if len(label_paths) != 2000:
        raise ValueError('Duplicate label stems')
    images, boxes, class_presence, hashes, clipped = [], [], [], [], 0
    out_of_range_annotations = []
    for i, member in enumerate(sorted(rgb_members, key=lambda m: m.filename)):
        p = dest / member.filename
        with Image.open(p) as image:
            image.load()
            width, height = image.size
            if image.mode != 'RGB':
                raise ValueError(f'Unexpected RGB mode: {p}: {image.mode}')
        images.append({'id': i + 1, 'file_name': member.filename, 'width': width, 'height': height})
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
        instance_boxes, present = [], np.zeros(12, dtype=np.int32)
        for line in label_paths[p.stem].read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            parts = list(map(float, line.split()))
            if len(parts) != 5 or not all(math.isfinite(v) for v in parts):
                raise ValueError(f'Malformed annotation: {p.stem}: {line}')
            cls, cx, cy, bw, bh = parts
            if cls != int(cls) or not 0 <= cls < 12 or bw <= 0 or bh <= 0:
                raise ValueError(f'Out-of-range annotation: {p.stem}: {line}')
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and bw <= 1 and bh <= 1):
                out_of_range_annotations.append({'image': member.filename, 'original_label': line,
                                                 'action': 'clip to image boundaries'})
            raw = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
            x1,y1,x2,y2 = [max(0, min(limit, v)) for v,limit in zip(raw,[width,height,width,height])]
            clipped += int(any(abs(a-b)>1e-5 for a,b in zip(raw,[x1,y1,x2,y2])))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Empty box: {p.stem}: {line}')
            instance_boxes.append({'image_id': i+1, 'category_id': int(cls),
                                   'bbox': [x1,y1,x2-x1,y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
            present[int(cls)] = 1
        boxes.append(instance_boxes)
        class_presence.append(present)
    split_path = annotations / 'split.json'
    filenames = [x['file_name'] for x in images]
    if len({Path(x).stem for x in filenames}) != 2000:
        raise ValueError('Duplicate RGB stems')
    if split_path.exists():
        split = json.loads(split_path.read_text())
        if split['filenames'] != filenames or split['sha256'] != hashes:
            raise ValueError('Dataset changed since saved split; refusing to overwrite it')
        train_ids, val_ids = split['train_indices'], split['val_indices']
    else:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=400, random_state=20260929)
        train, val = next(splitter.split(np.zeros((2000,1)), np.asarray(class_presence)))
        train_ids, val_ids = train.tolist(), val.tolist()
        # Multilabel stratification can deviate slightly from the requested size.
        # Correct the size deterministically while preserving all class coverage.
        presence = np.asarray(class_presence)
        target = presence.sum(0) * .2
        while len(val_ids) != 400:
            removing = len(val_ids) > 400
            source = val_ids if removing else train_ids
            counts = presence[val_ids].sum(0)
            source_counts = presence[source].sum(0)
            eligible = [i for i in source if np.all(source_counts - presence[i] > 0)]
            if not eligible:
                raise ValueError('Cannot achieve 1600/400 while preserving class coverage')
            direction = -1 if removing else 1
            chosen = min(eligible, key=lambda i: (float(np.sum(
                ((counts + direction * presence[i] - target) / np.maximum(target, 1))**2)), i))
            source.remove(chosen)
            (train_ids if removing else val_ids).append(chosen)
        train_ids, val_ids = sorted(train_ids), sorted(val_ids)
        split = {'seed': 20260929, 'method': 'multilabel stratification with deterministic exact-size correction',
                 'filenames': filenames, 'sha256': hashes, 'train_indices': train_ids, 'val_indices': val_ids}
    if len(train_ids) != 1600 or len(val_ids) != 400 or set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(range(2000)):
        raise ValueError('Invalid 1600/400 partition')
    if {hashes[i] for i in train_ids} & {hashes[i] for i in val_ids}:
        raise ValueError('Exact duplicate images cross train/validation; group split required')
    report = {'clipped_boxes': clipped, 'out_of_range_annotations': out_of_range_annotations,
              'split_seed': 20260929, 'subsets': {}}
    for name, ids in [('train1600',train_ids),('val400',val_ids),('train2000',list(range(2000)))]:
        anns = [dict(b, id=j+1) for j,b in enumerate(b for i in ids for b in boxes[i])]
        counts = Counter(a['category_id'] for a in anns)
        if len(counts) != 12:
            raise ValueError(f'{name} lacks classes: {set(range(12))-set(counts)}')
        coco = {'info': {'description': f'AICOMP official RGB {name}'}, 'licenses': [],
                'images': [images[i] for i in ids], 'annotations': anns,
                'categories': [{'id':i,'name':n} for i,n in enumerate(NAMES)]}
        (annotations / f'{name}.json').write_text(json.dumps(coco))
        (annotations / f'{name}.txt').write_text(''.join(filenames[i]+'\n' for i in ids))
        report['subsets'][name] = {'images':len(ids),'boxes':len(anns),'class_counts':{NAMES[i]:counts[i] for i in range(12)}}
    split_path.write_text(json.dumps(split, indent=2))
    (annotations / 'data_report.json').write_text(json.dumps(report, indent=2))
    backup = ROOT / 'backups/recovery-code-and-split.tar.gz'
    backup.parent.mkdir(exist_ok=True)
    with tarfile.open(backup, 'w:gz') as archive:
        for relative in ['scripts', 'configs', 'README.md', 'requirements-lock.txt',
                         'environment.yml', 'data/annotations', 'checkpoints/provenance.json',
                         'backups/dfine-source.bundle']:
            path = ROOT / relative
            if path.exists():
                archive.add(path, arcname=relative,
                            filter=lambda info: None if '__pycache__' in info.name else info)
    (annotations / 'READY').write_text('RGB image decode, annotation bounds, ZIP CRC and split overlap checks passed\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
