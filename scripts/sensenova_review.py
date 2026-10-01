"""Frozen D-FINE proposals, blind VLM region classification, validation only.

No model training, phase2 inputs, new boxes, class changes, GT-dependent proposal
selection, or score fitting. Fixed, bounded one-way score penalties are compared
on exactly the same fully reviewed images; other proposal scores stay unchanged.
Constrained single-token choices are conditional likelihoods, not confidence.
"""
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import string
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from sensenova_runtime import load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/sensenova_review'
ARCHIVE = ROOT/'初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip'
PREDICTIONS = ROOT/'experiments/mechanism_followup/bn_frozen800/raw_predictions.json'
ANNOTATIONS = ROOT/'data/annotations/val400.json'
RAM = Path('/dev/shm/aicomp_sensenova')
VARIANTS = ['rgb', 'rgb_ir', 'rgb_shuffled_ir']
SEED = 20261001
NAMES = ['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle','background','unknown']
PROMPT = ('The first image shows the RGB scene, with one candidate region marked by a red rectangle. '
          'The second image is a crop around that region. Classify the object inside the marked '
          'region, rather than other nearby objects. Do not locate new objects or output coordinates. '
          'Choose background if the region contains no object from the listed categories. Choose '
          'unknown if the object is too small, blurred, occluded, or ambiguous to identify reliably. '
          'Category definitions: seat means a chair, bench or stool; sign means a signboard; '
          'light means a lamp or light fixture, not bright sky; uav means an unmanned aerial drone; '
          'tricycle means a three-wheeled vehicle. ')
IR_PROMPT = ('The third image is the corresponding auxiliary infrared crop. Use it only as '
             'supporting evidence; it may be blurred or locally misaligned. The target is still '
             'the marked RGB region. ')

def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)); tmp.replace(path)

def select_indices(pred, iid):
    order = sorted(range(len(pred['scores'])),key=lambda k:-pred['scores'][k])[:100]
    rng = random.Random(SEED+iid)
    # Equal rank-stratum sampling, independent of annotations and VLM answers.
    picked = []
    for left,right in [(0,10),(10,30),(30,60),(60,100)]:
        pool = order[left:right]
        picked.extend(rng.sample(pool,min(3,len(pool))))
    return sorted(picked,key=lambda k:-pred['scores'][k])

def crop_bounds(box, size):
    x0,y0,x1,y1 = box; w,h = size
    bw,bh=max(x1-x0,1),max(y1-y0,1)
    ew,eh=max(bw*1.6,32),max(bh*1.6,32)
    cx,cy=(x0+x1)/2,(y0+y1)/2
    return (max(0,math.floor(cx-ew/2)),max(0,math.floor(cy-eh/2)),
            min(w,math.ceil(cx+ew/2)),min(h,math.ceil(cy+eh/2)))

def adjustment(record, original, strength):
    if record['abstain']:
        return float(original)
    probs=record['choice_probabilities']; own=record['dfine_label']
    top=record['choice_id']; margin=probs[top]-probs[own]
    # Unknown never vetoes. Correct/ambiguous recognition leaves scores intact.
    if top in (own,13) or margin < .2 or probs[own] > .2:
        return float(original)
    return float(original)*(1-strength*min(1.,max(0.,margin)))

def auc(labels, scores):
    from scipy.stats import rankdata
    labels=np.asarray(labels,dtype=bool); p=int(labels.sum()); n=len(labels)-p
    if not p or not n: return None
    ranks=rankdata(scores)
    return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))

def summarize(dataset,preds,plan,records):
    from faster_coco_eval import COCO, COCOeval_faster
    from torchvision.ops import box_iou
    index={(r['image_id'],r['box_index'],r['variant']):r for r in records}
    paired=[c for c in plan if all((c['image_id'],c['box_index'],v) in index for v in VARIANTS)]
    by_image={}
    for c in plan: by_image.setdefault(c['image_id'],[]).append(c)
    full_images=[i for i,cases in by_image.items() if all((i,c['box_index'],v) in index for c in cases for v in VARIANTS)]
    anns={i:[] for i in by_image}
    for a in dataset['annotations']:
        if a['image_id'] in anns and not a.get('iscrowd',0) and not a.get('ignore',0): anns[a['image_id']].append(a)
    diagnostics={}
    for v in VARIANTS:
        labels=[];supports=[];native=[];adjusted=[];tp_harmed=fp_penalized=abstains=0
        for c in paired:
            r=index[c['image_id'],c['box_index'],v]; targets=anns[c['image_id']]
            same=[a for a in targets if a['category_id']==r['dfine_label']]
            boxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in same]
            ious=box_iou(torch.tensor([r['box']],dtype=torch.float32),torch.tensor(boxes,dtype=torch.float32).reshape(-1,4))
            positive=bool(ious.numel() and ious.max()>=.5)
            original=float(r['dfine_score']); new=adjustment(r,original,.5)
            labels.append(positive);native.append(original);adjusted.append(new)
            supports.append(r['choice_probabilities'][r['dfine_label']] if not r['abstain'] else .5)
            tp_harmed+=int(positive and new<original);fp_penalized+=int(not positive and new<original);abstains+=int(r['abstain'])
        diagnostics[v]={'reviewed_paired_regions':len(paired),'tp_iou50':sum(labels),'fp_iou50':len(labels)-sum(labels),
                        'abstentions':abstains,'true_positive_penalized':tp_harmed,'false_positive_penalized':fp_penalized,
                        'native_score_auc':auc(labels,native),'vlm_support_auc':auc(labels,supports),
                        'penalty_half_score_auc':auc(labels,adjusted)}
    result={'paired_regions':len(paired),'paired_images':len(full_images),'diagnostics':diagnostics,
            'notes':['Validation-only pilot; no phase2 score, fitting or training.',
                     '12 rank-stratified candidates per image; unreviewed scores unchanged.',
                     'Token choice probabilities are conditional on 14 options; not calibrated confidence.',
                     'Diagnostic positives use same-class IoU>=.5, not one-to-one COCO matching.',
                     'Scores unchanged for tiny regions, low option mass, unknown or weak disagreement.']}
    if full_images:
        gt=COCO(); gt.dataset={**dataset,'images':[i for i in dataset['images'] if i['id'] in full_images],
                             'annotations':[a for a in dataset['annotations'] if a['image_id'] in full_images]};gt.createIndex()
        conditions=[('dfine_original',None,0)]+[(f'{v}_penalty_{s}',v,s) for v in VARIANTS for s in [.25,.5,1.]]
        evaluations={}
        for name,variant,strength in conditions:
            rows=[]
            for iid in full_images:
                pred=preds[str(iid)]
                for k in sorted(range(len(pred['scores'])),key=lambda k:-pred['scores'][k])[:100]:
                    score=float(pred['scores'][k]);r=index.get((iid,k,variant))
                    if r is not None: score=adjustment(r,score,strength)
                    b=pred['boxes'][k];rows.append({'image_id':iid,'category_id':pred['labels'][k],
                         'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':score})
            ev=COCOeval_faster(gt,gt.loadRes(rows),'bbox');ev.params.imgIds=full_images
            ev.evaluate();ev.accumulate();ev.summarize()
            evaluations[name]={'map50_95':float(ev.stats[0]*100),'ap50':float(ev.stats[1]*100),
                               'ap75':float(ev.stats[2]*100),'small_ap':float(ev.stats[3]*100)}
        result['evaluations']=evaluations
    write(OUT/'summary.json',result)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dataset=json.loads(ANNOTATIONS.read_text());preds=json.loads(PREDICTIONS.read_text())
    cohort=json.loads((ROOT/'experiments/sensenova_probe/identity.json').read_text())['image_ids']
    images={i['id']:i for i in dataset['images']}
    assert [c['name'] for c in dataset['categories']]==NAMES[:12]
    plan=[{'image_id':iid,'box_index':k} for iid in cohort for k in select_indices(preds[str(iid)],iid)]
    identity={'source_commit':'4366e0e1f4d22d2207ecc8e339f886b765ba876e',
              'model_revision':'79548fcc5b954598799b9317f8d3ec5e347d5c0e',
              'prediction_sha256':hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
              'annotation_sha256':hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
              'seed':SEED,'plan':plan,'variants':VARIANTS,'prompt':PROMPT,'ir_prompt':IR_PROMPT,
              'crop_pad':.3,'tiny_side_abstain':8,'min_option_mass':.05,
              'constrained_decoding':'14 shuffled single-token letter options, then EOS',
              'penalties':[.25,.5,1.],'no_score_fitting':True,'boxes_and_classes_fixed':True}
    if (OUT/'identity.json').exists(): assert json.loads((OUT/'identity.json').read_text())==identity
    else: write(OUT/'identity.json',identity)
    write(OUT/'plan.json',plan)
    record_path=OUT/'records.jsonl';records=[json.loads(s) for s in record_path.read_text().splitlines()] if record_path.exists() else []
    completed={(r['image_id'],r['box_index'],r['variant']) for r in records}
    write(OUT/'status.json',{'stage':'loading_model','time':time.time(),'completed_requests':len(records),'total_requests':len(plan)*len(VARIANTS)})
    import subprocess
    assert subprocess.check_output(['git','-C',str(RAM/'source'),'rev-parse','HEAD'],text=True).strip()==identity['source_commit']
    assert json.loads((ROOT/'experiments/sensenova_probe/model_identity.json').read_text())['revision']==identity['model_revision']
    model=load_model(RAM/'model',RAM/'source',OUT)
    option_tokens=[]
    for letter in string.ascii_uppercase[:14]:
        ids=model.tokenizer.encode(letter,add_special_tokens=False)
        assert len(ids)==1 and model.tokenizer.decode(ids)==letter
        option_tokens.append(ids[0])
    state={}
    def constrain(_,inputs,output):
        assert output.shape[0]==1
        masked=torch.full_like(output,-float('inf'))
        if not state:
            logits=output[0].float(); subset=logits[option_tokens]
            state['probabilities']=subset.softmax(-1).cpu().tolist()
            state['option_mass']=float(torch.exp(torch.logsumexp(subset,0)-torch.logsumexp(logits,0)))
            masked[0,option_tokens]=output[0,option_tokens]
        else:
            masked[0,model.new_token_ids['eos_token_id']]=0
        return masked
    hook=model.model.language_model.lm_head.register_forward_hook(constrain)
    with zipfile.ZipFile(ARCHIVE) as archive,record_path.open('a') as stream:
        members={(Path(p).parts[-2],Path(p).name):p for p in archive.namelist() if len(Path(p).parts)>1 and Path(p).parts[-2] in ['visible','infrared']}
        def load(modality,iid):
            with Image.open(io.BytesIO(archive.read(members[modality,Path(images[iid]['file_name']).name]))) as im: return im.convert('RGB')
        for ci,iid in enumerate(cohort):
            rgb=load('visible',iid);ir=load('infrared',iid)
            assert rgb.size==ir.size==(images[iid]['width'],images[iid]['height'])
            shuffled=load('infrared',cohort[(ci+1)%len(cohort)]).resize(rgb.size)
            pred=preds[str(iid)]
            for c in [c for c in plan if c['image_id']==iid]:
                k=c['box_index'];box=pred['boxes'][k];bounds=crop_bounds(box,rgb.size)
                assert bounds[2]>bounds[0] and bounds[3]>bounds[1]
                context=rgb.copy();ImageDraw.Draw(context).rectangle(box,outline='red',width=max(2,round(rgb.width/500)))
                crop=rgb.crop(bounds)
                order=list(range(14));random.Random(SEED+iid*1000+k).shuffle(order)
                options=' '.join(f'{string.ascii_uppercase[j]}: {NAMES[cat]}.' for j,cat in enumerate(order))
                for variant in VARIANTS:
                    if (iid,k,variant) in completed: continue
                    start=time.monotonic();state.clear()
                    write(OUT/'status.json',{'stage':'reviewing','image_id':iid,'box_index':k,'variant':variant,
                          'completed_requests':len(records),'total_requests':len(plan)*len(VARIANTS),'time':time.time()})
                    tiny=min(box[2]-box[0],box[3]-box[1])<8
                    prompt=PROMPT+(IR_PROMPT if variant!='rgb' else '')+options+' Answer with exactly one uppercase option letter and nothing else.'
                    if tiny:
                        choice=13;probs=[0.]*14;probs[13]=1.;mass=0.;text='SKIPPED_TINY_REGION'
                    else:
                        contents=[{'type':'image','value':context},{'type':'image','value':crop}]
                        if variant!='rgb': contents.append({'type':'image','value':(ir if variant=='rgb_ir' else shuffled).crop(bounds)})
                        contents.append({'type':'text','value':prompt})
                        for gpu in range(4): torch.cuda.reset_peak_memory_stats(gpu)
                        with torch.inference_mode():
                            text=model.generate(contents=contents,mode='understanding',noise_seed=SEED+iid+k,max_think_token_n=2)
                        letter=str(text).strip();assert letter in string.ascii_uppercase[:14],repr(text)
                        choice=order[string.ascii_uppercase.index(letter)];probs=[0.]*14
                        for j,cat in enumerate(order):probs[cat]=state['probabilities'][j]
                        mass=state['option_mass']
                    record={'image_id':iid,'box_index':k,'variant':variant,'box':box,'crop_bounds':bounds,
                            'dfine_label':int(pred['labels'][k]),'dfine_score':float(pred['scores'][k]),
                            'choice_id':choice,'choice_name':NAMES[choice],'choice_probabilities':probs,
                            'option_mass':mass,'abstain':bool(tiny or choice==13 or mass<.05),
                            'tiny_region':tiny,'raw_response':text,'prompt':prompt,'option_order':order,
                            'seconds':time.monotonic()-start,
                            'peak_reserved_gib':[torch.cuda.max_memory_reserved(g)/1024**3 for g in range(4)]}
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                    records.append(record);completed.add((iid,k,variant))
                    print(json.dumps({key:record[key] for key in ['image_id','box_index','variant','choice_name','option_mass','abstain','seconds','peak_reserved_gib']}),flush=True)
                    if len(completed)<=6:
                        preview=context.copy();preview.thumbnail((1280,720));preview.save(OUT/f'{iid}_{k}_context.jpg');crop.save(OUT/f'{iid}_{k}_crop.jpg')
                    torch.cuda.empty_cache()
                summarize(dataset,preds,plan,records)
    hook.remove();summarize(dataset,preds,plan,records)
    write(OUT/'status.json',{'stage':'complete','completed_requests':len(records),'total_requests':len(plan)*len(VARIANTS),'time':time.time()})
    (OUT/'COMPLETE').write_text('Frozen candidate region review pilot complete; no phase2 score\n')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);write(OUT/'status.json',{'stage':'failed','error':repr(exc),'time':time.time()});raise
