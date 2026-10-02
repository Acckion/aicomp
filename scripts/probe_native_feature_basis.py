"""Train-only feature compatibility diagnostic; no optimizer or test input."""
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import native_grid
from src.core import YAMLConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native_feature_basis'
OUT.mkdir(parents=True,exist_ok=True)


def main():
    lock=(ROOT/'experiments/mechanism_trials/gpu2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    assert torch.cuda.mem_get_info()[0]>=5120*1024**2
    cfg=YAMLConfig(str(ROOT/'configs/rgb1600.yml'),eval_spatial_size=[800,800])
    model=cfg.model;weights=native_grid.parent_weights(ROOT/'runs/ft_aug800/weights_epoch_020.pth')
    model.load_state_dict(weights,strict=True);del weights;model.cuda().eval()
    pool=json.loads((ROOT/'data/annotations/train1600.json').read_text());annotations=defaultdict(list)
    for a in pool['annotations']:annotations[a['image_id']].append(a)
    candidates=[im for im in pool['images'] if im['width']>=1280 and annotations[im['id']]]
    random.Random(20261002).shuffle(candidates);candidates=candidates[:32]
    records=[]
    def tensor(image):
        a=np.asarray(image.convert('RGB')).copy();return torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/255
    with torch.inference_mode():
        for index,im in enumerate(candidates):
            with Image.open(ROOT/'data/train'/im['file_name']) as image:
                orig=tensor(image);rgb=tensor(image.resize((800,800),Image.Resampling.BILINEAR))
            h,w=orig.shape[-2:];low=F.interpolate(rgb,size=(h,w),mode='bilinear',align_corners=False)
            with torch.autocast('cuda',dtype=torch.float16):
                global_feature=model.encoder(model.backbone(rgb))[0]
                ch,cw=global_feature.shape[-2:];coverage=global_feature.new_zeros((1,1,ch,cw))
                stitched={key:torch.zeros_like(global_feature) for key in ['raw_projected','regional_encoded','regional_low_encoded']}
                crop_h,crop_w=round(h*.6),round(w*.6)
                for y in (0,h-crop_h):
                    for x in (0,w-crop_w):
                        tile=F.interpolate(orig[:,:,y:y+crop_h,x:x+crop_w],size=(800,800),mode='bilinear',align_corners=False)
                        sham=F.interpolate(low[:,:,y:y+crop_h,x:x+crop_w],size=(800,800),mode='bilinear',align_corners=False)
                        backbone=model.backbone(tile)
                        regional={'raw_projected':model.encoder.input_proj[0](backbone[0]),'regional_encoded':model.encoder(backbone)[0], 'regional_low_encoded':model.encoder(model.backbone(sham))[0]}
                        ya,yb=round(y/h*ch),round((y+crop_h)/h*ch);xa,xb=round(x/w*cw),round((x+crop_w)/w*cw)
                        coverage[:,:,ya:yb,xa:xb]+=1
                        for key,feature in regional.items():stitched[key][:,:,ya:yb,xa:xb]+=F.interpolate(feature,size=(yb-ya,xb-xa),mode='bilinear',align_corners=False)
                assert (coverage>0).all()
                stitched={key:v/coverage for key,v in stitched.items()}
            masks={key:torch.zeros((ch,cw),device='cuda',dtype=torch.bool) for key in ['small_fg','other_fg','any_fg']}
            for a in annotations[im['id']]:
                x,y,bw,bh=a['bbox'];xa=max(0,min(cw-1,int(x/w*cw)));ya=max(0,min(ch-1,int(y/h*ch)))
                xb=min(cw,max(xa+1,int(np.ceil((x+bw)/w*cw))));yb=min(ch,max(ya+1,int(np.ceil((y+bh)/h*ch))))
                key='small_fg' if min(bw*800/w,bh*800/h)<16 else 'other_fg'
                masks[key][ya:yb,xa:xb]=True;masks['any_fg'][ya:yb,xa:xb]=True
            masks['background']=~(F.max_pool2d(masks['any_fg'][None,None].float(),3,stride=1,padding=1)[0,0]>0)
            record={'image_id':im['id'],'groups':{}}
            for group,mask in masks.items():
                if not mask.any():continue
                reference=global_feature.float()[0,:,mask];base_rms=reference.square().mean().sqrt().clamp_min(1e-8)
                d={}
                for key,feature in stitched.items():
                    v=feature.float()[0,:,mask];rolled=torch.roll(feature.float(),shifts=(17,43),dims=(-2,-1))[0,:,mask]
                    d[key]={'cosine':float(F.cosine_similarity(reference,v,dim=0).mean()),'shuffled_cosine':float(F.cosine_similarity(reference,rolled,dim=0).mean()),'rms_ratio':float(v.square().mean().sqrt()/base_rms)}
                delta=(stitched['regional_encoded']-stitched['regional_low_encoded']).float()[0,:,mask]
                d['encoded_native_minus_low']={'rms_ratio':float(delta.square().mean().sqrt()/base_rms)}
                record['groups'][group]=d
            records.append(record)
            (OUT/'status.json').write_text(json.dumps({'stage':'probing','images':index+1,'total':32}))
    groups={}
    for group in ['small_fg','other_fg','background']:
        rows=[r['groups'][group] for r in records if group in r['groups']]
        if rows:groups[group]={'images':len(rows),'mean':{key:{metric:float(np.mean([r[key][metric] for r in rows])) for metric in rows[0][key]} for key in rows[0]}}
    report={'training_images':32,'checkpoint':'ft_aug800 epoch20 EMA','groups':groups,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'limitations':'Feature cosine/RMS are compatibility descriptors, not detection AP, proof of informative native detail, or an automatic reason to train. GT only defines train-image diagnostic masks; no test input or optimization.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));(OUT/'records.json').write_text(json.dumps(records));(OUT/'status.json').write_text(json.dumps({'stage':'complete','images':32}));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
