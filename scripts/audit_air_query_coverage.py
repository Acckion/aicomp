"""Compare query geometry on official train images before spending training time.

No classification/score filtering: coverage is optimistic and is not AP or
one-to-one recall. Both policies share exactly one public-initialized model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'D-FINE')]
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert, box_iou
import taxonomy_air_queries as policy
import train_baseline as baseline
from mechanism_runtime import install_training_controls
from src.core import YAMLConfig
from src.zoo.dfine.dfine_decoder import DFINETransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(20260929)
    cfg = YAMLConfig(args.config)
    limit = cfg.yaml_cfg['gpu_memory_limit_gib']
    torch.cuda.set_per_process_memory_fraction(limit*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg.tuning = str(ROOT/'checkpoints/dfine_x_obj365.pth')
    install_training_controls(baseline)
    solver = baseline.BaselineSolver(cfg); solver._setup()
    model = solver.model.eval()
    data = json.loads((ROOT/'data/annotations/scene_train.json').read_text())
    heldout = json.loads((ROOT/'data/annotations/scene_val.json').read_text())
    train_ids = {im['id'] for im in data['images']}
    assert train_ids.isdisjoint(im['id'] for im in heldout['images'])
    annotations = defaultdict(list)
    for a in data['annotations']: annotations[a['image_id']].append(a)
    aerial = [i for i in sorted(train_ids) if any(a['category_id']==10 for a in annotations[i])]
    other = sorted(train_ids-set(aerial)); rng = random.Random(20260929)
    rng.shuffle(aerial); rng.shuffle(other)
    chosen = set(aerial[:32]+other[:32])
    records = []
    with torch.inference_mode():
        for im in data['images']:
            if im['id'] not in chosen: continue
            with Image.open(ROOT/'data/train'/im['file_name']) as image:
                x = TF.to_tensor(TF.resize(image.convert('RGB'), [800,800])).unsqueeze(0).cuda()
            anns = annotations[im['id']]
            boxes = torch.tensor([a['bbox'] for a in anns],dtype=torch.float32,device='cuda').reshape(-1,4)
            boxes = box_convert(boxes,'xywh','xyxy')/boxes.new_tensor([im['width'],im['height']]*2)
            results = {}
            for name, select in [('baseline',policy._original_select),('air_queries',policy.select_air_queries)]:
                DFINETransformer._select_topk = select
                with torch.autocast('cuda',dtype=torch.float16): pred = model(x)
                candidates = box_convert(pred['pred_boxes'][0].float(),'cxcywh','xyxy').clamp(0,1)
                assert len(candidates)==300 and candidates.isfinite().all()
                results[name] = box_iou(candidates,boxes).amax(0).cpu().tolist()
            for j,a in enumerate(anns):
                records.append({'image_id':im['id'],'annotation_id':a['id'],'class':a['category_id'],
                                'baseline_iou':results['baseline'][j],'air_queries_iou':results['air_queries'][j]})
            print('TRAIN_GEOMETRY',im['id'],len(records),flush=True)
    DFINETransformer._select_topk = policy.select_air_queries
    summary = {}
    for name, rows in [('uav',[r for r in records if r['class']==10]),('other',[r for r in records if r['class']!=10])]:
        summary[name]={'targets':len(rows)}
        for threshold in [.5,.75]:
            summary[name][str(threshold)]={key:sum(r[key+'_iou']>=threshold for r in rows) for key in ['baseline','air_queries']}
    uav = summary['uav']['0.5']; other_counts = summary['other']['0.5']
    supported = uav['air_queries']-uav['baseline']>=2 and other_counts['air_queries']>=other_counts['baseline']-max(2,int(summary['other']['targets']*.02))
    report={'images':len(chosen),'train_only':True,'public_initialization_only':True,
            'source_rows':list(policy.SOURCE_ROWS),'query_budget':300,'summary':summary,
            'launch_supported':supported,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'limitations':'Optimistic class-agnostic query geometry, not AP, one-to-one recall, or evidence of generalization. No heldout/test images or fitting.'}
    (out/'records.json').write_text(json.dumps(records)); (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
