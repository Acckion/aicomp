"""Label-backed error decomposition on the original independent val400 only."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from analyze_phase2_transfers import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/phase2_transfer_audit'


def main():
    dataset=json.loads((ROOT/'data/annotations/val400.json').read_text())
    names={c['id']:c['name'] for c in dataset['categories']}
    gt=defaultdict(list)
    for a in dataset['annotations']:
        x,y,w,h=a['bbox'];gt[a['image_id']].append((a['category_id'],[x,y,x+w,y+h]))
    grouped=json.loads((ROOT/'experiments/scene_groups/report.json').read_text())['groups']
    train={i['id'] for i in json.loads((ROOT/'data/annotations/train1600.json').read_text())['images']}
    shared=set().union(*(set(g) for g in grouped if set(g)&train))
    sources={
        'original_rgb1600_800':ROOT/'experiments/next_stage/infer800_tile0/raw_predictions.json',
        'ir1600_gateoff_800':ROOT/'experiments/tonight_20261002_validated_ir/validation_predictions.json',
    }
    result={}
    for model,path in sources.items():
        pred=json.loads(path.read_text());counts=defaultdict(lambda:defaultdict(int))
        for im in dataset['images']:
            p=pred[str(im['id'])];scores=np.asarray(p['scores']);order=np.argsort(-scores)[:100];order=order[scores[order]>=.25]
            boxes=np.asarray(p['boxes'],dtype=float).reshape(-1,4)[order];labels=np.asarray(p['labels'],dtype=int)[order]
            entries=gt[im['id']];truth=np.asarray([e[1] for e in entries],dtype=float).reshape(-1,4);cats=np.asarray([e[0] for e in entries],dtype=int)
            matrix=iou(boxes,truth);used=set()
            for k,c in enumerate(labels):
                counters=[counts[names[int(c)]],counts['ALL'],counts['shared_candidates' if im['id'] in shared else 'no_detected_link']]
                same=np.flatnonzero(cats==c);available=[j for j in same if j not in used and matrix[k,j]>=.5]
                if available:
                    j=max(available,key=lambda j:matrix[k,j]);used.add(int(j));typ='matched_correct_iou50'
                elif len(same) and matrix[k,same].max()>=.5:
                    typ='duplicate_correct_class_iou50'
                elif len(cats) and np.any(matrix[k,cats!=c]>=.5):
                    typ='overlap_other_class_iou50'
                elif len(same) and matrix[k,same].max()>=.1:
                    typ='localization_error_iou10_to50'
                else:
                    typ='no_same_class_geometric_support'
                for d in counters:d[typ]+=1;d['boxes_ge025']+=1
            for c in cats:
                counts[names[int(c)]]['gt_total']+=1
            counts['ALL']['gt_total']+=len(cats)
        result[model]={c:dict(v) for c,v in counts.items()}
    report={'score_threshold':.25,'iou_threshold':.5,'results':result,
        'limitations':'Independent val400 only, no test labels. Thresholded error counts are not AP; reference is original RGB1600 at800, not a matched continuation control. Geometric no-support can also reflect incomplete annotation; sample labels need inspection. Old train/val background grouping is descriptive.'}
    (OUT/'validation_errors.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{c:v for c,v in r.items() if c in ['ALL','garbage can','bicycle','sign','light']} for k,r in result.items()}))


if __name__=='__main__':main()
