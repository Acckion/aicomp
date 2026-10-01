"""Check annotation/feature alignment, positive-only gradients and native inference identity."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

import train_baseline as baseline
import vl_distillation
from src.core import YAMLConfig
from vl_distill_cache import write_json, sha256

ROOT=baseline.ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(20260929)
    config=YAMLConfig(ROOT/'configs/vl_distill800.yml')
    criterion=config.criterion.cuda()
    # An accepted matched query gets a nonzero gradient. A rejected teacher and
    # an unmatched/background query receive no semantic gradient.
    vector=torch.randn(1,3,768,device='cuda',requires_grad=True)
    target={'vl_features':torch.nn.functional.normalize(torch.randn(2,768,device='cuda'),dim=-1),
            'vl_valid':torch.tensor([True,False],device='cuda'),'labels':torch.tensor([0,6],device='cuda')}
    terms=criterion.distillation_losses({'vl_embeddings':vector},[target],[(torch.tensor([2,0]),torch.tensor([0,1]))],0)
    sum(terms.values()).backward()
    assert torch.isfinite(vector.grad).all()
    assert vector.grad[0,2].abs().sum()>0
    assert vector.grad[0,:2].abs().sum()==0
    empty={'vl_features':torch.zeros(0,768,device='cuda'),'vl_valid':torch.zeros(0,dtype=torch.bool,device='cuda'),'labels':torch.zeros(0,dtype=torch.long,device='cuda')}
    v=torch.randn(1,3,768,device='cuda',requires_grad=True)
    loss=criterion.distillation_losses({'vl_embeddings':v},[empty],[(torch.zeros(0,dtype=torch.long),torch.zeros(0,dtype=torch.long))],0)
    sum(loss.values()).backward();assert v.grad.abs().sum()==0
    loader=config.train_dataloader;loader.set_epoch(0)
    dataset=loader.dataset
    with np.load(ROOT/'experiments/vl_distillation/teacher/targets.npz') as values:
        rows={int(a):i for i,a in enumerate(values['annotation_ids'])}
        # Real augmentation/filtering must preserve the per-annotation cache.
        for index in range(24):
            image,t=dataset[index]
            rr=[rows[int(a)] for a in t['vl_annotation_ids']]
            assert np.array_equal(t['vl_features'].numpy(),values['features'][rr])
            assert np.array_equal(t['labels'].numpy(),values['labels'][rr])
            assert np.array_equal(t['vl_valid'].numpy(),values['valid'][rr])
            assert ((t['boxes']>=0)&(t['boxes']<=1)).all()
    detector=config.model
    parent=ROOT/'runs/ft_aug800/weights_epoch_020.pth'
    state=torch.load(parent,map_location='cpu',weights_only=False)
    weights=state['ema']['module'] if 'ema' in state else state['model']
    weights=dict(weights)
    for k in ('decoder.anchors','decoder.valid_mask'):weights[k]=detector.state_dict()[k]
    result=detector.load_state_dict(weights,strict=False)
    assert all(k.startswith('vl_projector.') for k in result.missing_keys) and not result.unexpected_keys
    # Deep-copied EMA hooks must point at the copied model, not the live model.
    clone=copy.deepcopy(detector)
    hook=list(clone.decoder.dec_score_head[-1]._forward_pre_hooks.values())[0]
    assert hook.__self__ is clone
    del clone
    native_cfg=YAMLConfig(ROOT/'configs/bn_frozen800.yml')
    native=native_cfg.model
    native.load_state_dict(weights,strict=True)
    detector=detector.cuda().eval();native=native.cuda().eval()
    image,_=dataset[0];image=image.unsqueeze(0).cuda()
    with torch.inference_mode():
        a=native(image);b=detector(image)
    difference={k:float((a[k]-b[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
    assert all(v==0 for v in difference.values()),difference
    assert 'vl_embeddings' not in b
    report={'stage':'passed','native_inference_max_abs_difference':difference,
            'gt_feature_alignment_augmented_images':24,'positive_only_gradient':True,
            'empty_targets_zero_gradient':True,'ema_hook_ownership':True,
            'parent_sha256':sha256(parent),'teacher_targets_sha256':sha256(ROOT/'experiments/vl_distillation/teacher/targets.npz'),
            'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    write_json(ROOT/'experiments/vl_distillation/preflight.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
