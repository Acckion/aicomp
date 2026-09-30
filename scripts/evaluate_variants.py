"""Offline, single-checkpoint resolution/slicing/postprocessing experiments."""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import batched_nms, box_iou
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'D-FINE'))
from src.core import YAMLConfig
import detail_encoder
from faster_coco_eval import COCO, COCOeval_faster


def select(pred, method='none', threshold=.6):
    boxes=torch.tensor(pred['boxes'],dtype=torch.float32).reshape(-1,4)
    scores=torch.tensor(pred['scores'],dtype=torch.float32)
    labels=torch.tensor(pred['labels'],dtype=torch.long)
    if method=='nms':
        keep=batched_nms(boxes,scores,labels,threshold)[:100]
    elif method=='soft':
        remaining=torch.arange(len(scores)); chosen=[]
        while len(remaining) and len(chosen)<100:
            j=scores[remaining].argmax(); idx=remaining[j]; chosen.append(idx.item())
            remaining=remaining[remaining!=idx]
            if not len(remaining): break
            iou=box_iou(boxes[idx:idx+1],boxes[remaining])[0]
            mask=(labels[remaining]==labels[idx]) & (iou>threshold)
            scores[remaining[mask]]*=1-iou[mask]
        keep=torch.tensor(chosen,dtype=torch.long)
    else: keep=scores.argsort(descending=True)[:100]
    return {'boxes':boxes[keep].tolist(),'scores':scores[keep].tolist(),'labels':labels[keep].tolist()}


def encode_txt(pred,w,h):
    lines=[]
    for box,score,label in zip(pred['boxes'],pred['scores'],pred['labels']):
        x1,y1,x2,y2=box
        lines.append(f'{label} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f} {score:.10f}')
    return '\n'.join(lines)+ ('\n' if lines else '')


def decode_txt(text,w,h):
    out={'boxes':[],'scores':[],'labels':[]}
    for line in text.splitlines():
        c,x,y,bw,bh,s=map(float,line.split())
        assert c==int(c) and 0<=c<12 and all(np.isfinite([x,y,bw,bh,s]))
        assert 0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1 and 0<=s<=1
        out['boxes'].append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        out['labels'].append(int(c));out['scores'].append(s)
    assert len(out['scores'])<=100
    return out


def coco_rows(preds):
    result=[]
    for iid,p in preds.items():
        for b,s,c in zip(p['boxes'],p['scores'],p['labels']):
            result.append({'image_id':int(iid),'category_id':int(c),'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':s})
    return result


def evaluate(gt,preds):
    ev=COCOeval_faster(gt,gt.loadRes(coco_rows(preds)),'bbox')
    ev.params.imgIds=sorted(map(int,preds));ev.evaluate();ev.accumulate();ev.summarize()
    per={}
    for i,c in enumerate(ev.params.catIds):
        p=ev.eval['precision'][:,:,i,0,-1];p=p[p>=0]
        per[gt.cats[c]['name']]=float(p.mean()*100) if p.size else None
    return {'map':float(ev.stats[0]*100),'stats':[float(v*100) for v in ev.stats],'per_class':per}


def windows(w,h,fraction):
    # Four overlapping views plus whole image; no model ensemble.
    cw,ch=round(w*fraction),round(h*fraction)
    return [(0,0,w,h)]+[(x,y,x+cw,y+ch) for y in (0,h-ch) for x in (0,w-cw)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--config',help='Model configuration for an architecture ablation');ap.add_argument('--size',type=int,default=640)
    ap.add_argument('--tile',type=float,default=0);ap.add_argument('--output',required=True);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--annotations',default=str(ROOT/'data/annotations/val400.json'))
    ap.add_argument('--image-root',default=str(ROOT/'data/train'))
    ap.add_argument('--predict-only',action='store_true')
    ap.add_argument('--gpu-memory-limit-gib',type=float,default=8.5)
    ap.add_argument('--method',choices=['none','nms','soft'],default='none')
    ap.add_argument('--threshold',type=float,default=.6)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert args.size % 32 == 0 and (args.tile == 0 or .5 <= args.tile < 1)

    torch.set_num_threads(2);torch.manual_seed(20260929)
    ann=Path(args.annotations); dataset=json.loads(ann.read_text()); images=dataset['images']
    gt=None if args.predict_only else COCO(str(ann))
    if args.limit:images=images[:args.limit]
    cache=out/'raw_predictions.json'
    identity={'checkpoint_sha256':hashlib.file_digest(open(args.checkpoint,'rb'),'sha256').hexdigest(),
              'annotations_sha256':hashlib.sha256(ann.read_bytes()).hexdigest(),
              'image_root':str(Path(args.image_root).resolve()),'size':args.size,'tile':args.tile,'limit':args.limit}
    if args.config:
        config = YAMLConfig(args.config)
        identity['resolved_model_config_sha256'] = hashlib.sha256(json.dumps(config.yaml_cfg,sort_keys=True,default=str).encode()).hexdigest()
    manifest=out/'cache_identity.json'
    if cache.exists():
        assert manifest.exists() and json.loads(manifest.read_text())==identity,'Stale prediction cache; use a new output directory'
    start=time.monotonic()
    if cache.exists():preds=json.loads(cache.read_text())
    else:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        cfg=YAMLConfig(args.config or str(ROOT/'configs/rgb1600.yml'),eval_spatial_size=[args.size,args.size])
        model=cfg.model
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        weights=state['ema']['module'] if 'ema' in state else state['model']
        # Spatial anchors are deterministic size-specific buffers, not learned weights.
        weights={k:v for k,v in weights.items() if k not in ('decoder.anchors','decoder.valid_mask')}
        missing,unexpected=model.load_state_dict(weights,strict=False)
        assert not unexpected and set(missing)<= {'decoder.anchors','decoder.valid_mask'},(missing,unexpected)
        model.cuda().eval();post=cfg.postprocessor;post.num_top_queries=300
        preds={};reference={}
        with torch.inference_mode():
            for im in images:
                image=Image.open(Path(args.image_root)/im['file_name']).convert('RGB');w,h=image.size
                boxes=[];scores=[];labels=[]
                views=windows(w,h,args.tile) if args.tile else [(0,0,w,h)]
                for x,y,x2,y2 in views:
                    crop=image.crop((x,y,x2,y2))
                    # Match validation's PIL bilinear resize and float32 inference.
                    tensor=TF.to_tensor(TF.resize(crop,[args.size,args.size])).unsqueeze(0).cuda()
                    p=post(model(tensor),torch.tensor([[x2-x,y2-y]],device='cuda'))[0]
                    if not args.tile:
                        reference[str(im['id'])]={k:p[k].cpu().tolist() for k in ('boxes','scores','labels')}
                    b=p['boxes'].cpu();b[:,[0,2]]+=x;b[:,[1,3]]+=y
                    b[:,[0,2]]=b[:,[0,2]].clamp(0,w);b[:,[1,3]]=b[:,[1,3]].clamp(0,h)
                    good=(b[:,2]>b[:,0]) & (b[:,3]>b[:,1])
                    boxes.extend(b[good].tolist());scores.extend(p['scores'].cpu()[good].tolist());labels.extend(p['labels'].cpu()[good].tolist())
                preds[str(im['id'])]={'boxes':boxes,'scores':scores,'labels':labels}
        cache.write_text(json.dumps(preds));manifest.write_text(json.dumps(identity,indent=2))
        if reference and not args.predict_only:
            (out/'unclipped_reference.json').write_text(json.dumps(reference))
    inference_seconds=time.monotonic()-start
    if args.predict_only:
        import zipfile
        folder=out/'prediction_txt';folder.mkdir(exist_ok=True)
        seen=set()
        with zipfile.ZipFile(out/'submission.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for im in images:
                filename=Path(im['file_name']).stem+'.txt'
                assert filename not in seen,'Duplicate submission filename'
                seen.add(filename)
                selected=select(preds[str(im['id'])],args.method,args.threshold)
                txt=encode_txt(selected,im['width'],im['height'])
                decode_txt(txt,im['width'],im['height'])
                (folder/filename).write_text(txt);archive.writestr(filename,txt)
        (out/'submission_manifest.json').write_text(json.dumps({**identity,'images':len(images),'method':args.method,'threshold':args.threshold,'validation':'all images present; finite valid coordinates, classes and confidence; at most 100 boxes/image','leaderboard_score':None},indent=2))
        (out/'COMPLETE').write_text('Prediction package complete; not submitted or scored\n')
        return
    reference_result=None
    if (out/'unclipped_reference.json').exists():
        raw=json.loads((out/'unclipped_reference.json').read_text())
        reference_result=evaluate(gt,{iid:select(p) for iid,p in raw.items()})
    results=[]
    for method,threshold in [('none',0),('nms',.5),('nms',.6),('nms',.7),('soft',.5),('soft',.7)]:
        selected={iid:select(p,method,threshold) for iid,p in preds.items()}
        row={'method':method,'threshold':threshold,**evaluate(gt,selected)}
        results.append(row)
        if method=='none':
            restored={}
            folder=out/'validation_txt';folder.mkdir(exist_ok=True)
            for im in images:
                iid=str(im['id']);txt=encode_txt(selected[iid],im['width'],im['height'])
                (folder/(Path(im['file_name']).stem+'.txt')).write_text(txt)
                restored[iid]=decode_txt(txt,im['width'],im['height'])
            roundtrip=evaluate(gt,restored)
            row['txt_roundtrip_map']=roundtrip['map'];row['txt_roundtrip_delta']=roundtrip['map']-row['map']
            assert abs(row['txt_roundtrip_delta'])<.01,row
    summary={'size':args.size,'tile_fraction':args.tile,'checkpoint':args.checkpoint,'images':len(images),'inference_seconds':inference_seconds,'unclipped_reference':reference_result,'results':results,'note':'Independent val400; scores are not leaderboard results. Full-image/crops use the same checkpoint.'}
    (out/'results.json').write_text(json.dumps(summary,indent=2));(out/'COMPLETE').write_text('ok\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
