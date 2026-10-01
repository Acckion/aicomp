"""Paired full-holdout memory ablation plus an equally trained control."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import train_baseline as baseline
import extra_iou_metrics
import instance_memory
from src.core import YAMLConfig
from pycocotools.cocoeval import COCOeval

ROOT=baseline.ROOT
def summarize(evaluator,records):
    coco=evaluator.coco_eval['bbox']
    counts=Counter(a['image_id'] for a in coco.cocoGt.anns.values() if not a.get('iscrowd',0))
    ids=[i for i in coco.params.imgIds if counts[i]>=10]
    if not ids:return {'dense_images':0,'dense_ap':None}
    # The upstream batched evaluator leaves cocoDt pointing at its final
    # batch. Reconstruct ALL detections, not that incomplete final object.
    detections=coco.cocoGt.loadRes(records)
    dense=COCOeval(coco.cocoGt,detections,'bbox');dense.params.imgIds=ids
    dense.evaluate();dense.accumulate();dense.summarize()
    return {'dense_images':len(ids),'dense_definition':'>=10 non-crowd GT instances per image',
            'dense_ap':float(dense.stats[0]),'dense_ap75':float(dense.stats[2])}

def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    instance_memory.install();results={}
    for name,run,enabled in [('memory_on','instance_memory800',True),('memory_off','instance_memory800',False),
                              ('disabled_control','instance_memory_control800',False)]:
        cfg=YAMLConfig(str(ROOT/'configs/instance_memory800.yml'))
        model=cfg.model
        path=ROOT/'runs'/run/'best.pth';state=torch.load(path,map_location='cpu',weights_only=False)
        weights=state['ema']['module'] if 'ema' in state else state['model']
        model.load_state_dict(weights,strict=True);model.cuda().eval()
        model.decoder.instance_memory.enabled=enabled
        evaluator=cfg.evaluator;records=[];original_update=evaluator.update
        def update(predictions):
            records.extend(evaluator.prepare(predictions,'bbox'))
            original_update(predictions)
        evaluator.update=update
        metrics,evaluator=baseline.evaluate(model,cfg.criterion.cuda(),cfg.postprocessor.cuda(),cfg.val_dataloader,
                                           evaluator,torch.device('cuda'),-1,False)
        assert len({r['image_id']for r in records})==400, 'Missing prediction images in paired evaluation'
        metrics.update(summarize(evaluator,records));metrics.update(checkpoint=str(path),best_epoch=state.get('best_epoch'),memory_enabled=enabled)
        results[name]=metrics
        (ROOT/'experiments/instance_memory/paired_report.json').write_text(json.dumps(results,indent=2))
        del model,cfg,evaluator;torch.cuda.empty_cache()
    results['deltas']={'memory_vs_disabled_control_ap':results['memory_on']['coco_eval_bbox'][0]-results['disabled_control']['coco_eval_bbox'][0],
                       'within_checkpoint_memory_effect_ap':results['memory_on']['coco_eval_bbox'][0]-results['memory_off']['coco_eval_bbox'][0]}
    (ROOT/'experiments/instance_memory/paired_report.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
