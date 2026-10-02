"""Paired scene-cluster bootstrap of cached COCO detections; CPU only.

Reuse per-image matching, verified by identity resampling, then resample
whole candidate background groups. Missing-class replicas are reported;
fixed12-class intervals are conditional on every class being represented.
This is descriptive uncertainty for two fixed checkpoints, not online AP,
training-seed uncertainty, or proof of physical scene independence.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib,io,json
from pathlib import Path
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def setup(annotations,predictions):
 with redirect_stdout(io.StringIO()):
  gt=COCO(annotations);dt=gt.loadRes(predictions);e=COCOeval(gt,dt,'bbox')
  e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.params.maxDets=[100]
  e.evaluate();e.accumulate()
 assert len(e.params.imgIds)==390
 base=list(e.evalImgs);params=deepcopy(e._paramsEval)
 precision=e.eval['precision'][:,:,:,0,0].copy()
 return e,base,params,precision


def resample(e,base,params,indices):
 count=len(params.imgIds);classes=len(params.catIds)
 e.params=deepcopy(params);e.params.imgIds=list(range(len(indices)));e._paramsEval=deepcopy(e.params)
 e.evalImgs=[base[k*count+i] for k in range(classes) for i in indices]
 with redirect_stdout(io.StringIO()):e.accumulate()
 precision=e.eval['precision'][:,:,:,0,0]
 valid=precision>=0
 per_class=[]
 for k in range(classes):
  p=precision[:,:,k];per_class.append(float(p[p>=0].mean()*100) if (p>=0).any() else None)
 return float(precision[valid].mean()*100),per_class


def main():
 p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--groups',required=True)
 p.add_argument('--pool',required=True);p.add_argument('--control',required=True);p.add_argument('--output',required=True)
 p.add_argument('--iterations',type=int,default=500);a=p.parse_args();assert a.iterations>=20
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 data=json.loads(Path(a.annotations).read_text());ids={im['id'] for im in data['images']};assert len(ids)==390
 group_data=json.loads(Path(a.groups).read_text())['groups']
 selected=[]
 for g in group_data:
  overlap=ids&set(g)
  if overlap:
   assert overlap==set(g),'Validation group split across train/val'
   selected.append(sorted(g))
 assert len([i for g in selected for i in g])==390 and {i for g in selected for i in g}==ids
 ee=[];identity=[]
 for source in [a.pool,a.control]:
  e,base,params,precision=setup(a.annotations,source);index={v:i for i,v in enumerate(params.imgIds)}
  score,classes=resample(e,base,params,list(range(390)))
  expected=float(precision[precision>=0].mean()*100)
  assert abs(score-expected)<1e-10,'Cached matching identity mismatch'
  ee.append((e,base,params));identity.append({'map':score,'per_class':classes,'identity_delta':score-expected})
 names={c['id']:c['name'] for c in data['categories']};class_names=[names[i] for i in ee[0][2].catIds]
 assert ee[0][2].imgIds==ee[1][2].imgIds and ee[0][2].catIds==ee[1][2].catIds
 groups=[[index[i] for i in g] for g in selected];rng=np.random.default_rng(20261003);rows=[]
 started=time.monotonic()
 with (out/'replicas.jsonl').open('w') as f:
  for n in range(a.iterations):
   draw=rng.integers(0,len(groups),len(groups));indices=[i for j in draw for i in groups[j]]
   scores=[resample(*v,indices) for v in ee]
   missing=[class_names[k] for k in range(len(class_names)) if scores[0][1][k] is None]
   assert all((scores[0][1][k] is None)==(scores[1][1][k] is None) for k in range(len(class_names)))
   r={'replica':n+1,'images':len(indices),'pool_map':scores[0][0],'control_map':scores[1][0],
      'delta':scores[0][0]-scores[1][0],'missing_classes':missing};rows.append(r);f.write(json.dumps(r)+'\n')
   if (n+1)%25==0:f.flush();print(json.dumps({'replicas':n+1,'seconds':round(time.monotonic()-started,1)}),flush=True)
 def interval(rr):
  d=np.array([r['delta'] for r in rr]);return {'replicas':len(rr),'median_delta':float(np.median(d)),
    'percentile_2_5_97_5':np.percentile(d,[2.5,97.5]).tolist(),'fraction_delta_positive':float((d>0).mean())} if len(d) else None
 deltas={name:identity[0]['per_class'][k]-identity[1]['per_class'][k] for k,name in enumerate(class_names)}
 report={'epoch':10,'images':390,'candidate_background_groups':len(groups),'iterations':a.iterations,'seed':20261003,
  'identity':identity,'point_delta':identity[0]['map']-identity[1]['map'],
  'per_class_delta':deltas,'leave_one_class_out_macro_delta':{name:float(np.mean([v for c,v in deltas.items() if c!=name])) for name in class_names},
  'all_replicas_descriptive':interval(rows),'fixed12_class_conditional':interval([r for r in rows if not r['missing_classes']]),
  'replicas_with_missing_classes':sum(bool(r['missing_classes']) for r in rows),
  'sources':{name:{'path':str(Path(path).resolve()),'sha256':digest(path)} for name,path in [('annotations',a.annotations),('groups',a.groups),('pool',a.pool),('control',a.control)]},
  'limitations':['Groups are geometric background candidates, not verified physical scenes or proof of independence.',
   'Fixed-checkpoint resampling does not measure training seed uncertainty, validation selection bias, or test distribution shift.',
   'All-replica COCO averages omit absent classes; fixed12 intervals condition on class presence and are not unconditional confidence guarantees.',
   'No test labels, new model training, prediction corrections, or submission generation.']}
 (out/'report.json').write_text(json.dumps(report,indent=2));(out/'COMPLETE').write_text('Paired descriptive bootstrap complete.\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
