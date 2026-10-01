"""Snapshot and audit candidate-review false vetoes without changing inference."""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.ops import box_iou

from sensenova_review import ROOT, OUT as REVIEW, ARCHIVE, ANNOTATIONS, PREDICTIONS, NAMES, VARIANTS, adjustment

OUT=ROOT/'experiments/sensenova_review/error_analysis'

def counterfactual(dataset,preds,paired,cases):
    """Oracle score restoration is for attribution only, never deployment."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    plan=json.loads((REVIEW/'plan.json').read_text())
    by_image=defaultdict(list)
    for c in plan:by_image[c['image_id']].append(c['box_index'])
    common=sorted(i for i,ks in by_image.items() if all((i,k) in paired for k in ks))
    if not common:return {'images':0}
    def evaluate(variant=None,restore=None):
        rows=[];mapping={};dt_id=0
        for iid in common:
            pred=preds[str(iid)]
            for k in sorted(range(len(pred['scores'])),key=lambda j:-pred['scores'][j])[:100]:
                b=pred['boxes'][k];score=pred['scores'][k]
                if variant and (iid,k) in paired and (restore is None or (iid,k) not in restore):
                    score=adjustment(paired[iid,k][variant],score,.5)
                rows.append({'image_id':iid,'category_id':pred['labels'][k],
                             'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':score})
                dt_id+=1;mapping[dt_id]=(iid,k)
        with redirect_stdout(io.StringIO()):
            gt=COCO();gt.dataset={**dataset,'images':[i for i in dataset['images'] if i['id'] in common],
                                 'annotations':[a for a in dataset['annotations'] if a['image_id'] in common]};gt.createIndex()
            ev=COCOeval(gt,gt.loadRes(rows),'bbox');ev.params.imgIds=common
            ev.evaluate();ev.accumulate();ev.summarize()
        matches={}
        for item in ev.evalImgs:
            if item is None or item['aRng']!=list(ev.params.areaRng[0]):continue
            for pos,did in enumerate(item['dtIds']):
                matches[mapping[did]]={'coco_tp_iou50':bool(item['dtMatches'][0,pos] and not item['dtIgnore'][0,pos]),
                                      'coco_tp_iou75':bool(item['dtMatches'][5,pos] and not item['dtIgnore'][5,pos])}
        return {'map50_95':float(ev.stats[0]*100),'ap75':float(ev.stats[2]*100)},matches
    baseline,matching=evaluate()
    result={'images':len(common),'image_ids':common,'baseline':baseline,'variants':{},
            'note':'GT-guided restoration is an oracle counterfactual for diagnosing score damage. It is not an executable scoring method or a validated gain.'}
    for c in cases:
        c['baseline_one_to_one_match']=matching.get((c['image_id'],c['box_index']))
    for variant in VARIANTS:
        score,_=evaluate(variant)
        harmed={(c['image_id'],c['box_index']) for c in cases if variant in c['harmed_variants'] and c['image_id'] in common}
        restored,_=evaluate(variant,harmed)
        item={'reviewed':score,'delta_vs_baseline':score['map50_95']-baseline['map50_95'],
              'oracle_restore_all_harmed':restored,'oracle_recovery_points':restored['map50_95']-score['map50_95'],
              'per_case_restoration':[]}
        for key in sorted(harmed):
            one,_=evaluate(variant,{key});item['per_case_restoration'].append({'image_id':key[0],'box_index':key[1],
                    'map_recovery_points':one['map50_95']-score['map50_95']})
        result['variants'][variant]=item
    return result

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('--records',type=Path,default=REVIEW/'records.jsonl');args=ap.parse_args()
    raw=args.records.read_bytes()
    # A live append may contain an incomplete final line. Snapshot complete rows.
    complete=raw[:raw.rfind(b'\n')+1]
    (OUT/'records_snapshot.jsonl').write_bytes(complete)
    records=[json.loads(s) for s in complete.splitlines()]
    dataset=json.loads(ANNOTATIONS.read_text());preds=json.loads(PREDICTIONS.read_text())
    images={i['id']:i for i in dataset['images']};anns=defaultdict(list)
    for a in dataset['annotations']:
        if not a.get('iscrowd',0) and not a.get('ignore',0):anns[a['image_id']].append(a)
    lookup=defaultdict(dict)
    for r in records:
        key=(r['image_id'],r['box_index']);assert r['variant'] not in lookup[key]
        lookup[key][r['variant']]=r
    paired={k:v for k,v in lookup.items() if len(v)==len(VARIANTS)}
    details=[];harm_cases=[]
    for (iid,k),variants in paired.items():
        r=variants['rgb'];meta=images[iid];targets=anns[iid]
        gtboxes=torch.tensor([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in targets],dtype=torch.float32).reshape(-1,4)
        ious=box_iou(torch.tensor([r['box']],dtype=torch.float32),gtboxes)[0]
        same=[j for j,a in enumerate(targets) if a['category_id']==r['dfine_label']]
        best=max(same,key=lambda j:float(ious[j])) if same else None
        overlap=float(ious[best]) if best is not None else 0.
        target=targets[best] if best is not None else None
        positive=overlap>=.5
        x0,y0,x1,y1=r['box'];short_native=min(x1-x0,y1-y0)
        # The reference evaluator stretches each RGB image to 800x800.
        # Scale width and height independently, rather than assuming letterbox.
        short800=min((x1-x0)*800/meta['width'],(y1-y0)*800/meta['height'])
        cx0,cy0,cx1,cy1=r['crop_bounds']
        in_crop=[a for a in targets if cx0 <= a['bbox'][0]+a['bbox'][2]/2 <= cx1 and cy0 <= a['bbox'][1]+a['bbox'][3]/2 <= cy1]
        other_classes=sorted(set(NAMES[a['category_id']] for a in in_crop if a['category_id']!=r['dfine_label']))
        base={'image_id':iid,'box_index':k,'category':NAMES[r['dfine_label']],
              'dfine_score':r['dfine_score'],'best_same_class_iou':overlap,
              'positive_iou50':positive,'positive_iou75':overlap>=.75,'gt_id':target['id'] if target else None,
              'native_short_side':short_native,'short_side800':short800,
              'size_bucket800':'under16' if short800<16 else '16to32' if short800<32 else '32plus',
              'centers_inside_crop':len(in_crop),'other_classes_in_crop':other_classes,
              'touches_image_border':bool(x0<=1 or y0<=1 or x1>=meta['width']-1 or y1>=meta['height']-1)}
        harm=[]
        for variant,v in variants.items():
            score=adjustment(v,v['dfine_score'],.5);penalized=score<float(v['dfine_score'])
            row={**base,'variant':variant,'choice':v['choice_name'],'p_own':v['choice_probabilities'][v['dfine_label']],
                 'p_choice':v['choice_probabilities'][v['choice_id']],'abstain':v['abstain'],
                 'adjusted_score_half':score,'score_reduction_half':float(v['dfine_score'])-score,
                 'penalized':penalized,'harmed_positive':bool(positive and penalized)}
            details.append(row)
            if row['harmed_positive']:harm.append(variant)
        if harm: harm_cases.append({**base,'box':r['box'],'crop_bounds':r['crop_bounds'],'gt':target,
                                    'harmed_variants':harm,'records':variants})
    stratified={}
    for variant in VARIANTS:
        rs=[r for r in details if r['variant']==variant];ps=[r for r in rs if r['positive_iou50']];hs=[r for r in ps if r['harmed_positive']]
        buckets={}
        for key in ['category','size_bucket800']:
            buckets[key]={}
            for value in sorted(set(r[key] for r in rs)):
                group=[r for r in rs if r[key]==value];pos=[r for r in group if r['positive_iou50']]
                harmed=[r for r in pos if r['harmed_positive']]
                buckets[key][value]={'regions':len(group),'positive_iou50':len(pos),'harmed_positive':len(harmed),
                                    'negative_penalized':sum(r['penalized'] and not r['positive_iou50'] for r in group)}
        stratified[variant]={'paired_regions':len(rs),'positive_iou50':len(ps),'harmed_positive':len(hs),
            'positive_iou75_harmed':sum(r['positive_iou75'] for r in hs),
            'background_vetoes_on_positives':sum(r['choice']=='background' for r in hs),
            'wrong_category_vetoes_on_positives':sum(r['choice']!='background' for r in hs),
            'harmed_original_score_ge08':sum(r['dfine_score']>=.8 for r in hs),
            'negative_penalized':sum(r['penalized'] and not r['positive_iou50'] for r in rs),
            'negative_penalized_original_score_ge05':sum(r['penalized'] and not r['positive_iou50'] and r['dfine_score']>=.5 for r in rs),
            'negative_penalized_median_original_score':float(np.median([r['dfine_score'] for r in rs if r['penalized'] and not r['positive_iou50']])) if any(r['penalized'] and not r['positive_iou50'] for r in rs) else None,
            'harmed_positive_median_original_score':float(np.median([r['dfine_score'] for r in hs])) if hs else None,
            'buckets':buckets}
    attribution=counterfactual(dataset,preds,paired,harm_cases)
    # Verify consequential AP numbers with the separate faster-COCO path used
    # by the live monitor. Redirect its output to this analysis folder only.
    import sensenova_review as review_module
    previous_output=review_module.OUT
    try:
        review_module.OUT=OUT
        with redirect_stdout(io.StringIO()):
            fast=review_module.summarize(dataset,preds,json.loads((REVIEW/'plan.json').read_text()),records)
    finally:review_module.OUT=previous_output
    differences=[abs(fast['evaluations']['dfine_original']['map50_95']-attribution['baseline']['map50_95'])]
    differences.extend(abs(fast['evaluations'][v+'_penalty_0.5']['map50_95']-attribution['variants'][v]['reviewed']['map50_95']) for v in VARIANTS)
    assert fast['paired_images']==attribution['images'] and max(differences)<1e-8
    report={'snapshot_time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'records_sha256':hashlib.sha256(complete).hexdigest(),'records':len(records),'paired_regions':len(paired),
            'sources':{'records':str(args.records),'annotations':str(ANNOTATIONS),'predictions':str(PREDICTIONS)},
            'definition':'Harmed positive: same-class best GT IoU>=.5 and registered half-strength penalty lowers its score; not necessarily COCO one-to-one TP or AP loss.',
            'stratified':stratified,'counterfactual_attribution':attribution,'unique_harmed_candidates':len(harm_cases),
            'verification':{'standard_vs_faster_coco_mAP_max_abs_difference':max(differences),
                            'rows_have_unique_image_box_variant_keys':True,'metric_images':attribution['images']},
            'harmed_cases':[{k:v for k,v in c.items() if k!='records'} for c in harm_cases],
            'limitations':['Live partial pilot snapshot; initial images ordered by area.',
                           '12 rank-stratified proposals per image, not all boxes or final validation.',
                           'True-positive diagnostic allows duplicate matches; inspect mAP before declaring damage.',
                           'GT overlays are analysis-only; never sent to VLM or used for test results.',
                           'Size and IR statistics are associations; crop/marker hypotheses require controlled inference.']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (OUT/'candidate_details.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]) if details else []);writer.writeheader();writer.writerows(details)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,14);small=ImageFont.truetype(fontpath,12)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members={(Path(p).parts[-2],Path(p).name):p for p in archive.namelist() if len(Path(p).parts)>1 and Path(p).parts[-2] in ['visible','infrared']}
        def load(modality,iid):
            with Image.open(io.BytesIO(archive.read(members[modality,Path(images[iid]['file_name']).name]))) as im:return im.convert('RGB')
        tiles=[]
        for case in harm_cases:
            iid,k=case['image_id'],case['box_index'];rgb=load('visible',iid);ir=load('infrared',iid)
            context=rgb.copy();draw=ImageDraw.Draw(context);draw.rectangle(case['box'],outline='red',width=max(2,rgb.width//400))
            context.thumbnail((420,250))
            crop=rgb.crop(case['crop_bounds']);rawcrop=crop.copy();ir_crop=ir.crop(case['crop_bounds'])
            x0,y0,x1,y1=case['box'];left,top,_,_=case['crop_bounds'];gt=case['gt'];gx,gy,gw,gh=gt['bbox']
            draw=ImageDraw.Draw(crop);draw.rectangle([x0-left,y0-top,x1-left,y1-top],outline='red',width=1)
            draw.rectangle([gx-left,gy-top,gx+gw-left,gy+gh-top],outline='#00ff00',width=1)
            tile=Image.new('RGB',(1280,360),'#181818');draw=ImageDraw.Draw(tile)
            title=f"Image {iid} / proposal {k}: {case['category']} | IoU {case['best_same_class_iou']:.3f} | score {case['dfine_score']:.3f} | native min {case['native_short_side']:.1f}px / at800 {case['short_side800']:.1f}px"
            draw.text((8,5),title,font=font,fill='white')
            panels=[(context,'RGB context; red = candidate'),(rawcrop,'Actual RGB crop sent to VLM'),(crop,'Audit overlay: red pred / green GT'),(ir_crop,'Actual corresponding IR crop')]
            for j,(im,label) in enumerate(panels):
                preview=im.copy();preview.thumbnail((305,245));x=j*320+(320-preview.width)//2
                tile.paste(preview,(x,40+(245-preview.height)//2));draw.text((j*320+7,287),label,font=small,fill='white')
            for j,v in enumerate(VARIANTS):
                r=case['records'][v];new=adjustment(r,r['dfine_score'],.5)
                draw.text((8,310+j*15),f"{v}: {r['choice_name']}; p(original)={r['choice_probabilities'][r['dfine_label']]:.3f}; score -> {new:.3f}; abstain={r['abstain']}",font=small,fill='orange' if v in case['harmed_variants'] else 'white')
            tile.save(OUT/f'case_{iid}_{k}.jpg',quality=90)
            tiles.append(tile)
        if tiles:
            montage=Image.new('RGB',(1280,len(tiles)*360),'white')
            for j,tile in enumerate(tiles):montage.paste(tile,(0,j*360))
            montage.save(OUT/'harmed_examples.jpg',quality=90)
    snapshot=OUT/'snapshots'/report['snapshot_time'][:19].replace('-','').replace(':','').replace('T','_')
    snapshot.mkdir(parents=True,exist_ok=True)
    for source in OUT.iterdir():
        if source.is_file() and source.name!='sources_receipt.json':shutil.copy2(source,snapshot/source.name)
    print(json.dumps({'snapshot_time':report['snapshot_time'],'records':len(records),'paired_regions':len(paired),
                      'unique_harmed_candidates':len(harm_cases),'stratified':stratified},ensure_ascii=False))

if __name__=='__main__':main()
