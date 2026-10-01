"""Evaluate stripped native checkpoints and predeclared val400 difficulty subsets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from faster_coco_eval import COCO, COCOeval_faster
from evaluate_variants import select, coco_rows

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=ROOT/'runs'/args.run
    output=ROOT/'experiments/aux_detection'/args.run/'best_native'
    output.mkdir(parents=True,exist_ok=True)
    state=torch.load(run/'best.pth',map_location='cpu',weights_only=False)
    source=state['ema']['module']if'ema'in state else state['model']
    checkpoint=run/'weights_best_native.pth'
    if not checkpoint.exists():
        temporary=checkpoint.with_suffix('.tmp')
        torch.save({'model':{k:v for k,v in source.items()if not k.startswith('aux_head.')},
                    'last_epoch':state['last_epoch'],'source':'EMA','auxiliary_head_removed':True},temporary)
        temporary.replace(checkpoint)
    del state, source
    command=[sys.executable,str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(checkpoint),
             '--size','800','--single-method','--method','none','--output',str(output)]
    subprocess.run(command,check=True,cwd=ROOT)
    predictions=json.loads((output/'raw_predictions.json').read_text())
    selected={iid:select(pred)for iid,pred in predictions.items()}
    gt=COCO(str(ROOT/'data/annotations/val400.json'))
    dense_ids=[iid for iid in gt.getImgIds()if len(gt.getAnnIds(imgIds=[iid]))>=10]
    tiny_ids=[]
    for iid in gt.getImgIds():
        image=gt.imgs[iid]
        annotations=gt.loadAnns(gt.getAnnIds(imgIds=[iid]))
        if any(min(a['bbox'][2]*800/image['width'],a['bbox'][3]*800/image['height'])<16
               for a in annotations):tiny_ids.append(iid)
    subsets={'all':gt.getImgIds(),'dense_ge10_gt':dense_ids,
             'non_dense_lt10_gt':sorted(set(gt.getImgIds())-set(dense_ids)),
             'contains_short_side_lt16_at800':tiny_ids}
    result={}
    detections=gt.loadRes(coco_rows(selected))
    for name,ids in subsets.items():
        evaluator=COCOeval_faster(gt,detections,'bbox');evaluator.params.imgIds=sorted(ids)
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        precision=evaluator.eval['precision']
        def average(values):
            values=values[values>=0]
            return float(values.mean()*100)if values.size else None
        result[name]={'images':len(ids),'map':float(evaluator.stats[0]*100),
                      'ap75':float(evaluator.stats[2]*100),'ap_small':float(evaluator.stats[3]*100),
                      'ap90':average(precision[8,:,:,0,-1])}
    report={'run':args.run,'checkpoint':str(checkpoint),'subsets':result,
            'protocol':'RGB800 native Top100, clipped boxes, no NMS/TTA/auxiliary head',
            'note':'Only independent val400; no phase2 claim. Subsets retain all categories/objects in selected images.'}
    (output/'difficulty_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
