"""Labeled-image GT-centered crop probe, separating zoom/context from new pixels.

These privileged crops diagnose available evidence. They are not a deployable
selector, detector ensemble, AP evaluation, or submission prediction path.
"""
import argparse,fcntl,hashlib,importlib,json,os,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert,box_iou
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'D-FINE'),str(ROOT/'scripts')]
from src.core import YAMLConfig

def crop_window(box,w,h,fraction):
    x,y,bw,bh=box
    cw=min(w,max(round(w*fraction),round(bw*1.5),1));ch=min(h,max(round(h*fraction),round(bh*1.5),1))
    left=max(0,min(w-cw,round(x+bw/2-cw/2)));top=max(0,min(h-ch,round(y+bh/2-ch/2)))
    return left,top,left+cw,top+ch

@torch.inference_mode()
def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/'experiments/mechanism_trials/gpu4.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=int(__import__('subprocess').check_output(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=3072
    torch.set_num_threads(2);torch.manual_seed(20261003);torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    cfg=YAMLConfig(args.config,eval_spatial_size=[800,800])
    for module in cfg.yaml_cfg.get('mechanism_imports',[]):importlib.import_module(module)
    model=cfg.model
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);assert 'ema' in state or state.get('source')=='EMA'
    weights=state['ema']['module'] if 'ema' in state else state['model'];weights={k:v for k,v in weights.items() if k not in ['decoder.anchors','decoder.valid_mask']}
    missing,extra=model.load_state_dict(weights,strict=False);assert not extra and set(missing)<={'decoder.anchors','decoder.valid_mask'}
    del state,weights
    model.cuda().eval()
    data=json.loads(Path(args.annotations).read_text());groups={i['id']:[] for i in data['images']}
    for gt in data['annotations']:
        if not gt.get('iscrowd',0):groups[gt['image_id']].append(gt)
    indices=torch.linspace(0,len(data['images'])-1,min(args.images,len(data['images']))).round().long().tolist()
    rows=[];sampled=0
    def infer(tensor,window):
        with torch.autocast('cuda',dtype=torch.float16):prediction=model(tensor)
        boxes=box_convert(prediction['pred_boxes'][0].float(),'cxcywh','xyxy')
        x,y,x2,y2=window;boxes=boxes*boxes.new_tensor([x2-x,y2-y,x2-x,y2-y])+boxes.new_tensor([x,y,x,y])
        return boxes,prediction['pred_logits'][0].float().sigmoid()
    def best(prediction,gt_box,label,threshold):
        boxes,scores=prediction;valid=(scores.argmax(-1)==label)&(scores[:,label]>=threshold)
        if not valid.any():return 0.
        return float(box_iou(boxes[valid],gt_box[None]).max())
    for index in indices:
        im=data['images'][index];w,h=im['width'],im['height']
        targets=[a for a in groups[im['id']] if min(a['bbox'][2]*800/w,a['bbox'][3]*800/h)<16]
        if not targets:continue
        image=Image.open(Path(args.image_root)/im['file_name']).convert('RGB');assert image.size==(w,h)
        global_rgb=TF.to_tensor(image.resize((800,800),Image.Resampling.BILINEAR))[None].cuda()
        native=TF.to_tensor(image)[None].cuda();low=F.interpolate(global_rgb,size=(h,w),mode='bilinear',align_corners=False)
        global_prediction=infer(global_rgb,(0,0,w,h));sampled+=1
        for a in targets:
            box=box_convert(torch.tensor(a['bbox'],device='cuda',dtype=torch.float32)[None],'xywh','xyxy')[0]
            box[0::2].clamp_(0,w);box[1::2].clamp_(0,h)
            if not (box[2]>box[0] and box[3]>box[1]):continue
            x,y,x2,y2=crop_window(box_convert(box[None],'xyxy','xywh')[0].tolist(),w,h,args.fraction)
            assert box[0]>=x and box[1]>=y and box[2]<=x2 and box[3]<=y2
            predictions={'global':global_prediction}
            for name,source in [('native_crop',native),('sham_crop',low)]:
                tensor=F.interpolate(source[:,:,y:y2,x:x2],size=(800,800),mode='bilinear',align_corners=False)
                predictions[name]=infer(tensor,(x,y,x2,y2))
            rows.append({'image_id':im['id'],'annotation_id':a['id'],'class':a['category_id'],'native_resolution':[w,h],'window':[x,y,x2,y2],'iou':{str(threshold):{name:best(pred,box,a['category_id'],threshold) for name,pred in predictions.items()} for threshold in [.05,.3]}})
        (out/'progress.json').write_text(json.dumps({'sampled_images_processed':sampled,'small_targets_processed':len(rows),'worker_pid':os.getpid()}))
    summary={}
    for cohort in ['all','high','low']:
        selected=[r for r in rows if cohort=='all' or (r['native_resolution']==[1920,1080])==(cohort=='high')]
        summary[cohort]={}
        for confidence in ['0.05','0.3']:
            summary[cohort][confidence]={'targets':len(selected),'counts':{name:{str(t):sum(r['iou'][confidence][name]>=t for r in selected) for t in [.5,.75,.9]} for name in ['global','native_crop','sham_crop']},'native_only_vs_sham':{str(t):sum(r['iou'][confidence]['native_crop']>=t>r['iou'][confidence]['sham_crop'] for r in selected) for t in [.5,.75,.9]},'sham_only_vs_native':{str(t):sum(r['iou'][confidence]['sham_crop']>=t>r['iou'][confidence]['native_crop'] for r in selected) for t in [.5,.75,.9]}}
    (out/'records.json').write_text(json.dumps(rows));report={'checkpoint_sha256':hashlib.file_digest(Path(args.checkpoint).open('rb'),'sha256').hexdigest(),'annotations_sha256':hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),'annotation_role':args.annotation_role,'labeled_images_sampled':len(indices),'images_with_small_targets':sampled,'crop_fraction':args.fraction,'summary':summary,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'limitations':'GT-centered privileged labeled crops; class-consistent candidates, per-GT max IoU, not one-to-one recall or AP. Different objects have different crop context. A positive result does not prove inference selection, distillation or phase2 benefit. Sham uses the same crop and zoom but only globally resized pixels.'}
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'COMPLETE').write_text('ok\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--config',required=True);parser.add_argument('--annotations',default=str(ROOT/'data/annotations/scene_train.json'));parser.add_argument('--image-root',default=str(ROOT/'data/train'));parser.add_argument('--output',required=True);parser.add_argument('--images',type=int,default=128);parser.add_argument('--fraction',type=float,default=.3)
    parser.add_argument('--annotation-role',choices=['training','validation'],default='training')
    args=parser.parse_args();assert args.images>0 and 0<args.fraction<=1;main(args)
