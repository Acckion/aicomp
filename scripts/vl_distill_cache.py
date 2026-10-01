"""Cache frozen CLIP targets from train1600 GT crops, never validation/test images."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    ['a photo of a person', 'a photo of a pedestrian'],
    ['a photo of a boat', 'a photo of a ship'],
    ['a photo of an animal', 'a photo of a dog', 'a photo of a cat', 'a photo of a bird'],
    ['a photo of a seat', 'a photo of a chair', 'a photo of a bench'],
    ['a photo of a sign', 'a photo of a traffic sign', 'a photo of a signboard'],
    ['a photo of a bicycle', 'a photo of a bike'],
    ['a photo of a car', 'a photo of a vehicle', 'a photo of a bumper car'],
    ['a photo of a ball', 'a photo of a sports ball'],
    ['a photo of a traffic light', 'a photo of a street lamp', 'a photo of a light fixture'],
    ['a photo of a garbage can', 'a photo of a trash bin'],
    ['a photo of a drone', 'a photo of an unmanned aerial vehicle'],
    ['a photo of a tricycle', 'a photo of a three wheeled vehicle'],
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


class CropDataset(Dataset):
    def __init__(self, annotations, images, image_root, processor, pad):
        self.annotations = annotations
        self.images = images
        self.root = Path(image_root)
        self.processor = processor
        self.pad = pad

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        meta = self.images[ann['image_id']]
        x, y, w, h = ann['bbox']
        with Image.open(self.root / meta['file_name']) as image:
            image = image.convert('RGB')
            crop = image.crop((max(0, math.floor(x-w*self.pad)), max(0, math.floor(y-h*self.pad)),
                               min(image.width, math.ceil(x+w*(1+self.pad))),
                               min(image.height, math.ceil(y+h*(1+self.pad)))))
            pixels = self.processor(images=crop, return_tensors='pt')['pixel_values'][0]
        return index, pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/vl_distillation/teacher')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'cache.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train_path = ROOT/'data/annotations/train1600.json'
    val_path = ROOT/'data/annotations/val400.json'
    train, val = json.loads(train_path.read_text()), json.loads(val_path.read_text())
    train_ids = {m['id'] for m in train['images']}
    assert len(train_ids)==1600 and len(val['images'])==400
    assert not train_ids.intersection(m['id'] for m in val['images'])
    assert not {m['file_name'] for m in train['images']}.intersection(m['file_name'] for m in val['images'])
    annotations = [a for a in train['annotations'] if not a.get('iscrowd',0) and min(a['bbox'][2:])>0]
    assert len({a['id'] for a in annotations}) == len(annotations)
    model_dir = ROOT/'checkpoints/clip/clip-vit-large-patch14-336'
    identity = {'train_annotations_sha256':sha256(train_path), 'validation_annotations_sha256':sha256(val_path),
                'teacher_weights_sha256':sha256(model_dir/'model.safetensors'),
                'teacher_config_sha256':sha256(model_dir/'config.json'),
                'teacher_processor_sha256':sha256(model_dir/'preprocessor_config.json'),
                'teacher':'openai/clip-vit-large-patch14-336', 'teacher_dtype':'float32',
                'pad':0.15,'prompts':PROMPTS, 'minimum_native_side':8,
                'feature_gate':'native short side>=8 and teacher top1 class agrees with training GT',
                'images':1600, 'annotations':len(annotations), 'validation_images_in_cache':0}
    identity_path = args.output/'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Existing teacher cache identity differs'
    else:
        write_json(identity_path, identity)
    if (args.output/'COMPLETE').exists():
        assert (args.output/'targets.npz').exists()
        return
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to('cuda').eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    with torch.inference_mode():
        tokens = processor(text=[p for group in PROMPTS for p in group], return_tensors='pt',padding=True,truncation=True)
        texts = model.get_text_features(**{k:v.cuda() for k,v in tokens.items()})
        texts = torch.nn.functional.normalize(texts.float(),dim=-1)
        prototypes=[]; start=0
        for group in PROMPTS:
            prototypes.append(torch.nn.functional.normalize(texts[start:start+len(group)].mean(0),dim=0))
            start += len(group)
        prototypes = torch.stack(prototypes).cpu().numpy()
    images = {m['id']:m for m in train['images']}
    dataset = CropDataset(annotations,images,ROOT/'data/train',processor,identity['pad'])
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True)
    features=np.zeros((len(annotations),model.config.projection_dim),dtype=np.float32)
    shards=args.output/'shards';shards.mkdir(exist_ok=True)
    done=np.zeros(len(annotations),dtype=bool)
    for file in sorted(shards.glob('batch_*.npz')):
        with np.load(file) as shard:
            ix=shard['indices'];features[ix]=shard['features'];done[ix]=True
    started=time.monotonic()
    for batch,(ix,pixels) in enumerate(loader):
        rows=ix.numpy()
        if done[rows].all():continue
        with torch.inference_mode():
            result=torch.nn.functional.normalize(model.get_image_features(pixel_values=pixels.cuda(non_blocking=True)).float(),dim=-1).cpu().numpy()
        file=shards/f'batch_{batch:05d}.npz';tmp=file.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,indices=rows,features=result)
        tmp.replace(file);features[rows]=result;done[rows]=True
        state={'stage':'caching','completed':int(done.sum()),'total':len(annotations),'gpu':args.gpu,
               'time':time.time(),'elapsed_seconds':time.monotonic()-started,
               'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
        write_json(args.output/'status.json',state)
        if batch%10==0:print(json.dumps(state),flush=True)
    assert done.all() and np.isfinite(features).all()
    labels=np.array([a['category_id'] for a in annotations],dtype=np.int64)
    short=np.array([min(a['bbox'][2:]) for a in annotations])
    sims=features@prototypes.T
    valid=(sims.argmax(-1)==labels)&(short>=8)
    path=args.output/'targets.npz'
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,annotation_ids=np.array([a['id'] for a in annotations],dtype=np.int64),
                            image_ids=np.array([a['image_id'] for a in annotations],dtype=np.int64),
                            labels=labels,features=features,valid=valid,prototypes=prototypes)
    path.with_suffix('.tmp').replace(path)
    audit={'annotations':len(annotations),'accepted_for_distillation':int(valid.sum()),
           'teacher_top1_agreement':float((sims.argmax(-1)==labels).mean()),
           'per_class':{train['categories'][c]['name']:{'objects':int((labels==c).sum()),
                       'accepted':int(((labels==c)&valid).sum())} for c in range(12)},
           'target_file_sha256':sha256(path),'identity':identity,
           'note':'Training-set diagnostic only. Agreement is a conservative feature gate, not validation accuracy.'}
    write_json(args.output/'audit.json',audit)
    write_json(args.output/'status.json',{'stage':'complete','completed':len(annotations),'total':len(annotations),'time':time.time(),'gpu':args.gpu})
    (args.output/'COMPLETE').write_text('Training-only frozen region embeddings complete.\n')
    print(json.dumps(audit),flush=True)


if __name__ == '__main__':
    main()
