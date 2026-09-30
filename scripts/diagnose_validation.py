"""Diagnose independent-val400 size/density cohorts using existing predictions only."""
import sys,json,contextlib,collections,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'D-FINE')]
import numpy as np
import torch
from torchvision.ops import box_iou
from evaluate_variants import select,evaluate
from faster_coco_eval import COCO


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',default=str(ROOT/'experiments/next_stage/infer800_tile0/raw_predictions.json'));ap.add_argument('--output',default=str(ROOT/'experiments/diagnostics'));ap.add_argument('--source',default='ft_aug800 epoch20, 800 plain, same independent val400');args=ap.parse_args()
 torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 ann=ROOT/'data/annotations/val400.json';data=json.loads(ann.read_text());gt=COCO(str(ann))
 raw=json.loads(Path(args.predictions).read_text())
 assert set(raw)=={str(i['id']) for i in data['images']},'Diagnosis requires the complete independent val400'
 pred={iid:select(p) for iid,p in raw.items()};by_image=collections.defaultdict(list)
 for a in data['annotations']:by_image[a['image_id']].append(a)
 counts={i['id']:len(by_image[i['id']]) for i in data['images']};dense_threshold=max(1,int(np.quantile(list(counts.values()),.75)))
 cohorts={'all':list(counts),'dense_top_quartile':[iid for iid,n in counts.items() if n>=dense_threshold],'less_dense':[iid for iid,n in counts.items() if n<dense_threshold],'small_heavy':[iid for iid,n in counts.items() if n>=3 and sum(a['area']<1024 for a in by_image[iid])/n>=.5]}
 report={'source':args.source,'dense_threshold_gt_boxes':dense_threshold,'cohorts':{},'classes':{},'notes':['Cohorts overlap; APs must not be averaged to reconstruct total.','Greedy error audit uses IoU .5 and score .05, separate from COCO AP; labels and coordinate units unchanged.','No phase2 ground truth or test training; no significance or domain-generalization claim.']}
 with (out/'evaluation.log').open('w') as log,contextlib.redirect_stdout(log):
  for name,ids in cohorts.items():
   if not ids:continue
   r=evaluate(gt,{str(i):pred[str(i)] for i in ids})
   report['cohorts'][name]={'images':len(ids),'gt_boxes':sum(counts[i] for i in ids),'small_gt_boxes':sum(a['area']<1024 for i in ids for a in by_image[i]),**r}
 errors={c['id']:collections.Counter() for c in data['categories']};difficult=[]
 for iid,anns in by_image.items():
  p=pred[str(iid)];keep=[j for j,s in enumerate(p['scores']) if s>=.05]
  pb=torch.tensor([p['boxes'][j] for j in keep],dtype=torch.float32).reshape(-1,4)
  gb=torch.tensor([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns],dtype=torch.float32).reshape(-1,4)
  ious=box_iou(pb,gb);matched=set();used=set()
  for j,original in enumerate(keep):
   eligible=[k for k,a in enumerate(anns) if k not in matched and a['category_id']==p['labels'][original] and ious[j,k]>=.5]
   if eligible:k=max(eligible,key=lambda k:float(ious[j,k]));matched.add(k);used.add(j)
  missed=0
  for k,a in enumerate(anns):
   e=errors[a['category_id']];e['gt']+=1;e['small_gt']+=int(a['area']<1024)
   if k in matched:e['covered']+=1;e['small_covered']+=int(a['area']<1024);continue
   missed+=1
   same=[j for j,original in enumerate(keep) if p['labels'][original]==a['category_id']]
   other=[j for j,original in enumerate(keep) if p['labels'][original]!=a['category_id']]
   best_same=max([float(ious[j,k]) for j in same],default=0)
   best_other=max([float(ious[j,k]) for j in other],default=0)
   if best_same>=.5:e['competition_or_duplicate_match']+=1
   elif best_other>=.5:e['class_confusion']+=1
   elif best_same>=.1:e['localization_or_partial_overlap']+=1
   else:e['missed_without_overlap']+=1
  difficult.append({'image_id':iid,'gt_boxes':len(anns),'uncovered_gt':missed,'small_gt_boxes':sum(a['area']<1024 for a in anns)})
 for c in data['categories']:
  e=dict(errors[c['id']]);e['coverage_at_score05_iou50']=e.get('covered',0)/e['gt'] if e.get('gt') else None
  report['classes'][c['name']]=e
 report['difficult_validation_images']=sorted(difficult,key=lambda x:x['uncovered_gt'],reverse=True)[:30]
 (out/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'cohorts':{n:{k:v[k] for k in ['images','gt_boxes','map']} for n,v in report['cohorts'].items()},'classes':report['classes']},ensure_ascii=False))


if __name__=='__main__':main()
