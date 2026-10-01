"""Measure IR-only detection and GT coverage complementary to fixed RGB.

The GT coverage union is an offline diagnostic, never a deployed ensemble.
No combined predictions or test submission files are written.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF

import train_baseline as baseline
import zip_ir_dataset
from src.core import YAMLConfig
from evaluate_variants import select,evaluate
from faster_coco_eval import COCO

ROOT=Path(__file__).resolve().parents[1]

def matched_gt(pred,annotations,threshold):
    keep=[i for i,s in enumerate(pred['scores']) if s>=.05]
    if not keep or not annotations:return set()
    p=torch.tensor([pred['boxes'][i] for i in keep],dtype=torch.float32).reshape(-1,4)
    g=torch.tensor([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]
                    for a in annotations],dtype=torch.float32).reshape(-1,4)
    iou=box_iou(p,g).numpy()
    same=np.array([pred['labels'][i] for i in keep])[:,None]==np.array([a['category_id']for a in annotations])[None,:]
    eligible=same&(iou>=threshold)
    pi,gi=linear_sum_assignment(eligible.astype(float)*1000+iou*eligible,maximize=True)
    return {annotations[j]['id']for i,j in zip(pi,gi) if eligible[i,j]}

def coverage_report(rgb,ir,data):
    by_image=collections.defaultdict(list)
    for a in data['annotations']:by_image[str(a['image_id'])].append(a)
    all_gt={a['id']:a for a in data['annotations']}
    result={}
    for threshold in [.5,.75,.9]:
        rgb_ids=set();ir_ids=set()
        for image in data['images']:
            iid=str(image['id']);anns=by_image[iid]
            rgb_ids.update(matched_gt(rgb[iid],anns,threshold))
            ir_ids.update(matched_gt(ir[iid],anns,threshold))
        unique=ir_ids-rgb_ids
        per_class={}
        for category in data['categories']:
            ids={k for k,a in all_gt.items() if a['category_id']==category['id']}
            per_class[category['name']]={'gt':len(ids),'rgb_covered':len(ids&rgb_ids),
                 'ir_covered':len(ids&ir_ids),'ir_unique':len(ids&unique)}
        result[str(threshold)]={'gt':len(all_gt),'rgb_covered':len(rgb_ids),'ir_covered':len(ir_ids),
             'ir_unique':len(unique),'ir_unique_small':sum(all_gt[k]['area']<1024 for k in unique),
             'gt_coverage_union_diagnostic':len(rgb_ids|ir_ids)/len(all_gt),
             'per_class':per_class,'ir_unique_annotation_ids':sorted(unique)}
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    config=YAMLConfig(str(ROOT/'configs/ir_detector800.yml'))
    model=config.model;state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    weights=state['ema']['module'] if 'ema' in state else state['model']
    model.load_state_dict(weights,strict=True);model.cuda().eval()
    data=json.loads((ROOT/'data/annotations/val400.json').read_text())
    if args.limit:data['images']=data['images'][:args.limit]
    dataset=zip_ir_dataset.ZipIRCocoDetection(str(ROOT/'data/train'),str(ROOT/'data/annotations/val400.json'),None,
         str(ROOT/'初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip'))
    checkpoint_sha=hashlib.file_digest(open(args.checkpoint,'rb'),'sha256').hexdigest()
    ir={};post=config.postprocessor;post.num_top_queries=100
    with torch.inference_mode():
        for index,image in enumerate(data['images']):
            pixels=dataset.load_ir(image)
            tensor=TF.to_tensor(TF.resize(pixels,[800,800])).unsqueeze(0).cuda()
            p=post(model(tensor),torch.tensor([[image['width'],image['height']]],device='cuda'))[0]
            boxes=p['boxes'].cpu()
            boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,image['width'])
            boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,image['height'])
            good=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
            ir[str(image['id'])]={'boxes':boxes[good].tolist(),'labels':p['labels'].cpu()[good].tolist(),
                               'scores':p['scores'].cpu()[good].tolist()}
            if (index+1)%50==0:print(f'IR validation predictions {index+1}/{len(data["images"])}',flush=True)
    if args.limit:
        ids={i['id']for i in data['images']};data['annotations']=[a for a in data['annotations'] if a['image_id']in ids]
    rgb_raw=json.loads((ROOT/'experiments/light_tuning/infer_plain/raw_predictions.json').read_text())
    rgb={str(i['id']):select(rgb_raw[str(i['id'])])for i in data['images']}
    gt=COCO(str(ROOT/'data/annotations/val400.json'))
    report={'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_sha256':checkpoint_sha,
            'validation_images':len(data['images']),'ir_map':evaluate(gt,ir),'fixed_rgb_map':evaluate(gt,rgb),
            'coverage':coverage_report(rgb,ir,data),
            'notes':['Independent val400; models trained only on train1600.',
                     'Coverage: same class, score>=.05, maximum-cardinality one-to-one matching per modality.',
                     'GT coverage union is diagnostic, not an achievable mAP or allowed deployed detection ensemble.',
                     'RGB-coordinate GT reused for IR; equal image sizes do not prove geometric registration.',
                     'No phase2 pixels/labels/training and no combined prediction or submission files.']}
    (out/'ir_raw_predictions.json').write_text(json.dumps(ir))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'COMPLETE').write_text('IR-only validation and complementary GT coverage complete\n')
    print(json.dumps({'ir_map':report['ir_map']['map'],'rgb_map':report['fixed_rgb_map']['map'],
                     'ir_unique_gt':{k:v['ir_unique']for k,v in report['coverage'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
