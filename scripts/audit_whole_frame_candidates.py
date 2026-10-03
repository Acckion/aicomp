"""Training-only candidate diagnostics; provisional-box overlap is not AP.

Capture the unchanged encoder selection and compare its300 candidates to larger
score-ranked pools. GT enters measurements only, never candidate selection.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import types
import torch
from torchvision.ops import box_iou
import train_baseline
import taxonomy_pooling
import whole_frame_head
from src.core import YAMLConfig

ROOT=Path(__file__).resolve().parents[1]


def xyxy(boxes):
    return torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-index',type=int,default=6)
    parser.add_argument('--sample',type=int,default=64);args=parser.parse_args()
    out=ROOT/'experiments/whole_frame_candidates_e1';out.mkdir(parents=True,exist_ok=True)
    owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/f'experiments/mechanism_trials/gpu{args.gpu_index}.lock').open('a')
    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
    data=json.loads((ROOT/'data/annotations/scene_train.json').read_text())
    by_id={im['id']:im for im in data['images']};small=set()
    for ann in data['annotations']:
        im=by_id[ann['image_id']];_,_,w,h=ann['bbox']
        if min(w*800/im['width'],h*800/im['height'])<16:small.add(im['id'])
    rng=random.Random(20261003);nsmall=min(len(small),args.sample*3//4)
    ids=rng.sample(sorted(small),nsmall)
    remaining=sorted(set(by_id)-set(ids));ids+=rng.sample(remaining,min(args.sample-len(ids),len(remaining)))
    records=[];provenance={}
    with torch.inference_mode():
        for mode in ['real','sham']:
            name='scene_whole_head_'+mode;cfg=YAMLConfig(str(ROOT/'configs'/f'{name}.yml'))
            dataset=cfg.train_dataloader.dataset;dataset.training=False
            model=cfg.model;checkpoint=ROOT/'runs'/name/'weights_epoch_001.pth'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['source']=='EMA'
            model.load_state_dict(state['model'],strict=True);del state
            provenance[mode]={'checkpoint':str(checkpoint),'sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()}
            model.cuda().eval();capture={};original=model.decoder._select_topk
            def hook(self,memory,logits,anchors,k):
                capture['memory']=memory;capture['logits']=logits;capture['anchors']=anchors
                return original(memory,logits,anchors,k)
            model.decoder._select_topk=types.MethodType(hook,model.decoder)
            assert model.decoder.query_select_method=='default'
            for step,image_id in enumerate(ids):
                packed,targets=cfg.train_dataloader.collate_fn([dataset[dataset.ids.index(image_id)]])
                packed=packed.cuda();target=targets[0];im=by_id[image_id]
                gt=target['boxes'].as_subclass(torch.Tensor).float()/torch.tensor([im['width'],im['height'],im['width'],im['height']])
                labels=target['labels'].cpu();gwh=gt[:,2:]-gt[:,:2]
                with torch.autocast('cuda',dtype=torch.float16):
                    final=model(packed)
                    logits=capture['logits'][0];anchors=capture['anchors'][0]
                    order=logits.max(-1).values.topk(min(3000,len(logits))).indices
                    proposed=(model.decoder.enc_bbox_head(capture['memory'][:,order])[0]+anchors[order]).sigmoid()
                boxes=xyxy(proposed.float()).cpu();enc_logits=logits[order].float().cpu().sigmoid()
                grid_centers=anchors[order,:2].sigmoid().float().cpu()
                ious=box_iou(boxes,gt);enc_labels=enc_logits.argmax(-1)
                # Final predictions use the same native flattened top100 rule.
                final_scores=final['pred_logits'][0].float().sigmoid().flatten()
                final_order=final_scores.topk(100).indices
                final_labels=(final_order%12).cpu();final_indices=final_order//12
                final_boxes=xyxy(final['pred_boxes'][0,final_indices].float()).cpu()
                final_ious=box_iou(final_boxes,gt)
                for j in range(len(gt)):
                    c=int(labels[j]);metrics={}
                    for budget in [300,1500,3000]:
                        same=(enc_labels[:budget]==c)&(enc_logits[:budget,c]>=.05)
                        metrics[str(budget)]={'any_iou':float(ious[:budget,j].max()),
                            'same_class_iou':float(ious[:budget,j][same].max()) if same.any() else 0.,
                            'grid_center_inside_gt':bool(((grid_centers[:budget]>=gt[j,:2])&(grid_centers[:budget]<=gt[j,2:])).all(-1).any())}
                    mask=final_labels==c
                    records.append({'mode':mode,'image_id':image_id,'gt_index':j,'category_id':c,
                        'small_shortside800':bool((gwh[j]*800).min()<16),
                        'coco_small_original_area':bool(gwh[j].prod()*im['width']*im['height']<1024),
                        'budgets':metrics,'final_top100_same_class_iou':float(final_ious[:,j][mask].max()) if mask.any() else 0.})
                capture.clear();del final,packed,proposed,logits,anchors,order
                (out/'progress.json').write_text(json.dumps({'mode':mode,'images_done':step+1,'worker_pid':os.getpid()}))
            model.decoder._select_topk=original;del model,cfg,dataset
            torch.cuda.empty_cache()
    summary={}
    for mode in ['real','sham']:
        summary[mode]={}
        for cohort in ['all','small_shortside800','coco_small_original_area']:
            rr=[r for r in records if r['mode']==mode and (cohort=='all' or r[cohort])]
            summary[mode][cohort]={'targets':len(rr),'budgets':{str(k):{
                'any_iou50':sum(r['budgets'][str(k)]['any_iou']>=.5 for r in rr),
                'same_class_iou50':sum(r['budgets'][str(k)]['same_class_iou']>=.5 for r in rr),
                'same_class_iou75':sum(r['budgets'][str(k)]['same_class_iou']>=.75 for r in rr),
                'grid_center_inside_gt':sum(r['budgets'][str(k)]['grid_center_inside_gt'] for r in rr)} for k in [300,1500,3000]},
                'final_top100_iou50':sum(r['final_top100_same_class_iou']>=.5 for r in rr),
                'final_top100_iou75':sum(r['final_top100_same_class_iou']>=.75 for r in rr)}
    report={'epoch':1,'training_images':len(ids),'small_enriched_images':nsmall,'sample_ids':ids,
        'provenance':provenance,'summary':summary,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'limitations':['Deliberately small-target enriched training sample, not heldout AP or representative recall.',
            'Provisional encoder boxes and grid centers can be refined by later decoder layers.',
            'Per-GT maximum overlap is optimistic, not one-to-one matching.',
            '1500/3000 are diagnostic pools, not deployed decoder budgets or proof of achievable gains.',
            'No model outputs, weights, training or test data changed.']}
    (out/'records.json').write_text(json.dumps(records)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'COMPLETE').write_text('ok\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
