"""Same single model and RGB pictures; fixed paired/shuffled/zero IR controls."""
import argparse
import json
from pathlib import Path
import torch
import ir_query_alignment
import extra_iou_metrics
import train_baseline as baseline
from src.core import YAMLConfig

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    cfg=YAMLConfig(str(ROOT/'configs/ir_query800.yml'))
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=cfg.model.cuda().eval();state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    weights=state['ema']['module']if'ema'in state else state['model']
    model.load_state_dict(weights,strict=True)
    loader=cfg.val_dataloader;dataset=loader.dataset
    names=sorted(Path(m['file_name']).name for m in dataset.coco.dataset['images'])
    original=dict(dataset.members)
    # Permutation may contain accidental fixed points: deterministic rotate
    # avoids any valid same-image pairing while retaining only val400 IR.
    shuffled=names[137:]+names[:137]
    assert all(a!=b for a,b in zip(names,shuffled))
    results={}
    for condition in ['paired','shuffled','zero']:
        dataset.members=dict(original)
        if condition=='shuffled':dataset.members.update({name:original[other]for name,other in zip(names,shuffled)})
        # A shuffled image may originate at another native size. Dataset keeps
        # RGB geometry/GT and independently resizes IR; no IR-coordinate GT.
        dataset.allow_shuffled_size=condition=='shuffled'
        model.ir_mode='zero'if condition=='zero'else'paired'
        metrics,_=baseline.evaluate(model,cfg.criterion.cuda(),cfg.postprocessor,loader,cfg.evaluator,'cuda',0,False)
        results[condition]=metrics
        output=ROOT/'experiments/ir_query_alignment/ablations.json'
        output.write_text(json.dumps({'checkpoint':args.checkpoint,'fixed_val_images':400,
                                     'shuffle':'sorted validation IR filenames rotated by 137; no training IR',
                                     'results':results},indent=2))
        print('IR_QUERY_ABLATION',condition,json.dumps(metrics),flush=True)

if __name__=='__main__':main()
