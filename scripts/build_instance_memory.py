"""Training-only multi-prototype regional memory in D-FINE's feature space.

GT RoIs supply class labels, without CLIP agreement filtering. Background
candidates are regions outside dilated GT, not independently verified labels.
No validation/test image contributes to the memory.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align
from torchvision.transforms import functional as TF

import train_baseline
from src.core import YAMLConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/instance_memory'
STORE=ROOT/'checkpoints/gpu6_storage/instance_memory'

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def status(stage,**kwargs):
    p=OUT/'cache_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'stage':stage,'time':time.time(),'pid':os.getpid(),**kwargs},indent=2));tmp.replace(p)

def cluster(features,count):
    """Spherical k-means with deterministic farthest-point initialization."""
    x=F.normalize(features.float(),dim=-1);count=min(count,len(x))
    centers=[x[0]];similarity=x@centers[0]
    for _ in range(1,count):
        centers.append(x[int(similarity.argmin())]);similarity=torch.maximum(similarity,x@centers[-1])
    centers=torch.stack(centers)
    for _ in range(15):
        assigned=(x@centers.T).argmax(1)
        centers=F.normalize(torch.stack([x[assigned==i].mean(0) if (assigned==i).any() else centers[i] for i in range(count)]),dim=-1)
    return centers

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    OUT.mkdir(parents=True,exist_ok=True);STORE.mkdir(parents=True,exist_ok=True)
    locks=[]
    for p in [OUT/'cache.lock',ROOT/f'experiments/mechanism_trials/gpu{args.gpu}.lock']:
        lock=p.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    try:
        if (STORE/'COMPLETE').exists():raise RuntimeError('Refusing to overwrite completed memory')
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
        train_path=ROOT/'data/annotations/train1600.json';val_path=ROOT/'data/annotations/val400.json'
        train=json.loads(train_path.read_text());val=json.loads(val_path.read_text())
        ids={x['id']for x in train['images']};assert len(ids)==1600 and not ids&{x['id']for x in val['images']}
        parent=ROOT/'runs/ft_aug800/weights_epoch_020.pth'
        config=YAMLConfig(str(ROOT/'configs/contextfull800.yml'));model=config.model
        state=torch.load(parent,map_location='cpu',weights_only=False);weights=dict(state['model'])
        current=model.state_dict()
        for key in ['decoder.anchors','decoder.valid_mask']:weights[key]=current[key]
        model.load_state_dict(weights,strict=True);model.cuda().eval()
        by_image=defaultdict(list)
        for a in train['annotations']:
            if not a.get('iscrowd',0):by_image[a['image_id']].append(a)
        images=sorted(train['images'],key=lambda x:x['id'])
        if args.limit:images=images[:args.limit]
        vectors=[];spatial=[];labels=[];ann_ids=[];image_ids=[];geometry=[]
        rng=np.random.default_rng(20261002)
        with torch.inference_mode():
            for i,meta in enumerate(images):
                with Image.open(ROOT/'data/train'/meta['file_name']) as image:
                    tensor=TF.to_tensor(TF.resize(image.convert('RGB'),[800,800])).unsqueeze(0).cuda()
                features=model.encoder(model.backbone(tensor))[0].float()
                boxes=[];anns=[]
                for a in by_image[meta['id']]:
                    x,y,w,h=a['bbox'];left=max(0,x)*800/meta['width'];top=max(0,y)*800/meta['height']
                    right=min(meta['width'],x+w)*800/meta['width'];bottom=min(meta['height'],y+h)*800/meta['height']
                    if right>left and bottom>top:boxes.append([left,top,right,bottom]);anns.append(a)
                gt=torch.tensor(boxes,device='cuda',dtype=torch.float32).reshape(-1,4)
                # Keep local spatial structure; centers alone lose boundary context.
                if len(gt):
                    region=roi_align(features,[gt],output_size=2,spatial_scale=features.shape[-1]/800,sampling_ratio=2,aligned=True)
                    for a,box,f in zip(anns,gt.cpu(),region.cpu()):
                        vectors.append(f.mean((-1,-2)));spatial.append(f.flatten());labels.append(a['category_id']);ann_ids.append(a['id']);image_ids.append(meta['id'])
                        geometry.append([(box[2]-box[0])/800,(box[3]-box[1])/800])
                # These candidate negatives follow the detector's complete-GT
                # assumption; their reliability must not be overstated.
                candidates=[]
                for _ in range(40):
                    x,y=rng.integers(0,736,size=2);candidates.append([x,y,x+64,y+64])
                bg=torch.tensor(candidates,device='cuda',dtype=torch.float32)
                if len(gt):
                    expanded=gt.clone();pad=(gt[:,2:]-gt[:,:2])*.25;expanded[:,:2]-=pad;expanded[:,2:]+=pad
                    bg=bg[box_iou(bg,expanded).max(1).values==0][:3]
                else:bg=bg[:3]
                if len(bg):
                    regions=roi_align(features,[bg],output_size=2,spatial_scale=features.shape[-1]/800,sampling_ratio=2,aligned=True).cpu()
                    for f in regions:
                        vectors.append(f.mean((-1,-2)));spatial.append(f.flatten());labels.append(12);ann_ids.append(-1);image_ids.append(meta['id']);geometry.append([.08,.08])
                if (i+1)%25==0 or i==0:
                    status('extracting',gpu=args.gpu,images=i+1,total_images=len(images),regions=len(vectors),peak_mib=torch.cuda.max_memory_allocated()/1024**2)
                    print(f'memory images={i+1}/{len(images)} regions={len(vectors)}',flush=True)
        features=torch.stack(vectors);label=torch.tensor(labels);centers=[];center_labels=[];counts={}
        for c in range(13):
            selected=features[label==c];counts[str(c)]=len(selected)
            if len(selected):
                prototypes=cluster(selected,16 if c==12 else 8);centers.append(prototypes);center_labels.extend([c]*len(prototypes))
        identity={'train_annotations_sha256':sha(train_path),'val_annotations_sha256':sha(val_path),'parent_sha256':sha(parent),'images':len(images),
                  'class_regions':counts,'feature_dim':features.shape[-1],'prototypes_per_class_max':8,'background_prototypes_max':16,
                  'source':'Frozen D-FINE-X encoder stride8 GT RoIAlign 2x2; training images only; no CLIP/text filtering',
                  'background_note':'Outside 25%-dilated GT candidate negatives; not independently verified annotations',
                  'validation_in_memory':False,'test_in_memory':False}
        name='smoke_memory.npz' if args.limit else 'memory.npz'
        np.savez_compressed(STORE/name,features=F.normalize(features,dim=-1).numpy(),spatial_features=torch.stack(spatial).numpy(),labels=label.numpy(),
                            annotation_ids=np.array(ann_ids),image_ids=np.array(image_ids),box_wh=np.array(geometry),
                            prototypes=torch.cat(centers).numpy(),prototype_labels=np.array(center_labels))
        (STORE/('smoke_identity.json'if args.limit else'identity.json')).write_text(json.dumps(identity,indent=2))
        if not args.limit:(STORE/'COMPLETE').write_text('Training-only memory ready\n')
        status('smoke_complete' if args.limit else 'complete',gpu=args.gpu,**identity);print(json.dumps(identity),flush=True)
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        for lock in locks:lock.close()

if __name__=='__main__':main()
