"""Measure final-layer/GO-union assignment disagreement on augmented train views."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'D-FINE'),str(ROOT/'scripts')]
import torch
from torchvision.ops import box_convert,box_iou
from src.core import YAMLConfig
from src.misc import dist_utils
import target_views


def match(criterion,outputs,targets):
    return criterion.matcher({k:outputs[k].float() for k in ('pred_logits','pred_boxes')},targets)['indices']


@torch.no_grad()
def main(batches):
    torch.set_num_threads(2);dist_utils.setup_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=YAMLConfig(str(ROOT/'configs/ft_aug800_shared3.yml'))
    cfg.yaml_cfg['train_dataloader']['total_batch_size']=2
    cfg.yaml_cfg['train_dataloader']['num_workers']=2
    model=cfg.model
    state=torch.load(ROOT/'runs/ft_aug800/weights_epoch_020.pth',map_location='cpu',weights_only=False)['model']
    model.load_state_dict(state,strict=True);model.cuda().train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    criterion=cfg.criterion.cuda()
    loader=cfg.train_dataloader;loader.set_epoch(0);loader.collate_fn.scales=[800]
    stats={'images':0,'final_positive_queries':0,'go_queries':0,'different_gt':0,'different_class':0,'matched_iou_under05':0,'matched_iou_under075':0}
    ious=[];short_ious=[]
    for index,(samples,targets) in enumerate(loader):
        if index>=batches:break
        samples=samples.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
        with torch.autocast('cuda'):outputs=model(samples,targets=targets)
        final=match(criterion,outputs,targets)
        aux=[match(criterion,o,targets) for o in outputs['aux_outputs']+[outputs['pre_outputs']]+outputs['enc_aux_outputs']]
        go=criterion._get_go_indices(final,aux)
        for b,(f,g) in enumerate(zip(final,go)):
            stats['images']+=1;stats['final_positive_queries']+=len(f[0]);stats['go_queries']+=len(g[0])
            gm=dict(zip(g[0].tolist(),g[1].tolist()))
            for q,t in zip(f[0].tolist(),f[1].tolist()):
                if gm[q]!=t:
                    stats['different_gt']+=1
                    stats['different_class']+=int(targets[b]['labels'][gm[q]]!=targets[b]['labels'][t])
            predicted=box_convert(outputs['pred_boxes'][b,f[0]].float(),'cxcywh','xyxy')
            truth=targets[b]['boxes'][f[1]]
            values=box_iou(predicted,box_convert(truth,'cxcywh','xyxy')).diag()
            ious.extend(values.tolist());short_ious.extend(values[truth[:,2:].min(1).values*800<16].tolist())
            stats['matched_iou_under05']+=int((values<.5).sum());stats['matched_iou_under075']+=int((values<.75).sum())
    stats['different_gt_fraction']=stats['different_gt']/max(1,stats['final_positive_queries'])
    stats['different_class_fraction']=stats['different_class']/max(1,stats['final_positive_queries'])
    stats['mean_matched_iou']=sum(ious)/len(ious) if ious else None
    stats['short_side_under16_matched_count']=len(short_ious)
    stats['short_side_under16_mean_iou']=sum(short_ious)/len(short_ious) if short_ious else None
    stats['notes']=['Training1600 views with existing augmentation, not val400 AP or test.',
                    'Training-mode decoder for auxiliary assignments; BN statistics held fixed; no gradients or checkpoint updates.',
                    'Assignment disagreement is a mechanism diagnostic, not proof of a faulty matching algorithm.']
    out=ROOT/'experiments/mechanism_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'matching_report.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batches',type=int,default=16);args=parser.parse_args();main(args.batches)
