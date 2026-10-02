"""Check that train-only source subtype evidence actually exists before training."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import random,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'D-FINE')]
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert
import taxonomy_subtype_dn
import train_baseline as baseline
from mechanism_runtime import install_training_controls
from src.core import YAMLConfig


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.manual_seed(20260929)
 cfg=YAMLConfig(a.config);limit=cfg.yaml_cfg.get('gpu_memory_limit_gib',6)
 torch.cuda.set_per_process_memory_fraction(limit*1024**3/torch.cuda.get_device_properties(0).total_memory)
 cfg.tuning=str(ROOT/'checkpoints/dfine_x_obj365.pth');install_training_controls(baseline)
 solver=baseline.BaselineSolver(cfg);solver._setup();model=solver.model.train()
 data=json.loads((ROOT/'data/annotations/scene_train.json').read_text());val=json.loads((ROOT/'data/annotations/scene_val.json').read_text())
 assert {im['id'] for im in data['images']}.isdisjoint(im['id'] for im in val['images'])
 by_image=defaultdict(list)
 for gt in data['annotations']:by_image[gt['image_id']].append(gt)
 images=[im for im in data['images'] if by_image[im['id']]];random.Random(20260929).shuffle(images);images=images[:64]
 records=[];sources=Counter()
 with torch.inference_mode():
  for im in images:
   with Image.open(ROOT/'data/train'/im['file_name']) as f:x=TF.to_tensor(TF.resize(f.convert('RGB'),[800,800])).unsqueeze(0).cuda()
   gt=by_image[im['id']];boxes=torch.tensor([r['bbox'] for r in gt],dtype=torch.float32,device='cuda')
   boxes=box_convert(boxes,'xywh','cxcywh')/boxes.new_tensor([im['width'],im['height']]*2)
   labels=torch.tensor([r['category_id'] for r in gt],dtype=torch.long,device='cuda')
   with torch.autocast('cuda',dtype=torch.float16):prediction=model(x,[{'boxes':boxes,'labels':labels}])
   assert prediction['pred_logits'].isfinite().all() and prediction['pred_boxes'].isfinite().all()
   stats=model.decoder.subtype_dn_stats;assert stats is not None and stats['targets']==len(gt)
   assert not hasattr(model.decoder.denoising_class_embed,'_subtype_context')
   records.append({'image_id':im['id'],**stats});sources.update(stats['source_rows'])
   print(json.dumps(records[-1]),flush=True)
 supported=sum(r['supported'] for r in records);total=sum(r['targets'] for r in records)
 report={'images':len(images),'targets':total,'supported':supported,'support_fraction':supported/total,'source_rows':dict(sources),
         'launch_supported':supported>=16 and len(sources)>=2,'public_source_only':True,'train_only':True,
         'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
         'limitations':'GT-compatible source subtype support is not verified fine-class correctness, AP improvement, or evidence of generalization. No heldout/test images or fitting.'}
 (out/'records.json').write_text(json.dumps(records));(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
