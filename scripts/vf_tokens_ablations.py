"""Single trained checkpoint: spatial feature interventions on independent val400."""
import argparse,json
from pathlib import Path
import torch
import vf_tokens_adapter,extra_iou_metrics
import train_baseline as baseline
from src.core import YAMLConfig
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=YAMLConfig(str(ROOT/'configs/vf_tokens800.yml'));cfg.yaml_cfg['val_dataloader']['total_batch_size']=2;model=cfg.model.cuda().eval()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'],strict=True);del state
    results={}
    for condition in ('real','zero','shuffle','wrong_image','disabled'):
        model.vf_mode='real' if condition=='disabled' else condition;model.vf_enabled=condition!='disabled';torch.manual_seed(20261002)
        metrics,_=baseline.evaluate(model,cfg.criterion.cuda(),cfg.postprocessor,cfg.val_dataloader,cfg.evaluator,'cuda',0,False)
        results[condition]=metrics
        (ROOT/'experiments/vf_tokens/ablations.json').write_text(json.dumps({'checkpoint':args.checkpoint,'results':results,'fixed_val_images':400,'intervention':'current-image patch features at true positions versus zero or same-image spatial permutation; wrong image via batch2 roll; no retraining'},indent=2))
        print('VF_ABLATION',condition,json.dumps(metrics),flush=True)
if __name__=='__main__':main()
