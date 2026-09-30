"""Local DEIMv2 screening with explicit class reset and continuous schedule.

Uses the pinned official DEIMv2 architecture/loss/data/evaluation and the
existing D-FINE training loop, which supports gradient accumulation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/model_sources/DEIMv2'),str(ROOT/'D-FINE')]
import torch
import yaml
from safetensors.torch import load_file
from engine.core import YAMLConfig
from engine.solver._solver import BaseSolver
from engine.solver.det_engine import evaluate
from engine.misc import dist_utils
from src.solver.det_engine import train_one_epoch
from schedule import WarmupCosine


def save(state,path):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);tmp.replace(path)


class Solver(BaseSolver):
    def train(self):
        # The upstream scheduler factory does not accept null scheduler configs.
        # This runner owns the continuous per-update warmup/cosine schedule.
        self._setup()
        self.optimizer=self.cfg.optimizer
        self.train_dataloader=self.cfg.train_dataloader
        self.val_dataloader=self.cfg.val_dataloader
        self.evaluator=self.cfg.evaluator

    def load_tuning_state(self,path):
        weights=load_file(path,device='cpu')
        current=self.model.state_dict()
        matched={};reset=[]
        for key,value in current.items():
            if key in ('decoder.anchors','decoder.valid_mask'):
                matched[key]=value
            elif key in ('decoder.up','decoder.reg_scale'):
                matched[key]=weights['decoder.decoder.'+key.split('.')[-1]]
            elif key not in weights:
                raise RuntimeError(f'Missing pretrained tensor: {key}')
            elif value.shape!=weights[key].shape:
                if 'score_head' not in key and 'denoising_class_embed' not in key:
                    raise RuntimeError(f'Unexpected shape mismatch: {key}')
                reset.append(key);matched[key]=value
            else:matched[key]=weights[key]
        unexpected=set(weights)-set(current)
        assert not unexpected,unexpected
        self.model.load_state_dict(matched,strict=True)
        print('COCO classification/denoising tensors reset for AICOMP12:',reset,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);dist_utils.setup_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cfg=YAMLConfig(a.config)
    torch.cuda.set_per_process_memory_fraction(cfg.yaml_cfg.get('gpu_memory_limit_gib',8.0)*1024**3/torch.cuda.get_device_properties(0).total_memory)
    out=Path(cfg.output_dir)
    if not a.smoke and (out/'metrics.jsonl').exists():raise RuntimeError('Partial run requires explicit resume support; refusing overwrite')
    solver=Solver(cfg);solver.train()
    accumulation=cfg.yaml_cfg['gradient_accumulation_steps'];epochs=cfg.yaml_cfg['epoches']
    steps=math.ceil(len(solver.train_dataloader)/accumulation)
    scheduler=WarmupCosine(solver.optimizer,epochs*steps,2*steps,.1)
    solver.lr_warmup_scheduler=scheduler
    (out/'resolved_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg,sort_keys=False))
    print('Pretrained SHA256:',hashlib.file_digest(open(cfg.tuning,'rb'),'sha256').hexdigest(),flush=True)
    print('Training batches/epoch:',len(solver.train_dataloader),'effective batch:',cfg.train_dataloader.batch_size*accumulation,flush=True)
    if a.smoke:
        solver.train_dataloader.set_epoch(0)
        peak=max(solver.train_dataloader.collate_fn.scales or [640])
        solver.train_dataloader.collate_fn.scales=[peak]
        batches=iter(solver.train_dataloader);data=[next(batches) for _ in range(accumulation*2)]
        train_one_epoch(solver.model,solver.criterion,data,solver.optimizer,solver.device,0,False,
                        max_norm=.1,ema=solver.ema,scaler=solver.scaler,
                        gradient_accumulation_steps=accumulation,lr_warmup_scheduler=scheduler,print_freq=1)
        evaluate(solver.ema.module,solver.criterion,solver.postprocessor,[next(iter(solver.val_dataloader))],solver.evaluator,solver.device)
        print('SMOKE TEST PASSED peak_size',peak,'peak_MiB',torch.cuda.max_memory_allocated()/1024**2,flush=True)
        return
    best=-1;best_epoch=None
    for epoch in range(epochs):
        start=time.monotonic();solver.train_dataloader.set_epoch(epoch)
        train=train_one_epoch(solver.model,solver.criterion,solver.train_dataloader,solver.optimizer,solver.device,epoch,False,
                              epochs=epochs,max_norm=.1,ema=solver.ema,scaler=solver.scaler,
                              gradient_accumulation_steps=accumulation,lr_warmup_scheduler=scheduler,print_freq=25)
        solver.last_epoch=epoch
        validation,evaluator=evaluate(solver.ema.module,solver.criterion,solver.postprocessor,solver.val_dataloader,solver.evaluator,solver.device)
        ap50_95=validation['coco_eval_bbox'][0]
        if not math.isfinite(ap50_95):raise RuntimeError('Nonfinite validation AP')
        per={}
        ev=evaluator.coco_eval['bbox']
        for i,catid in enumerate(ev.params.catIds):
            precision=ev.eval['precision'][:,:,i,0,-1];valid=precision[precision>=0]
            per[evaluator.coco_gt.cats[catid]['name']]=float(valid.mean()) if valid.size else None
        validation['per_class_ap']=per
        state=solver.state_dict();state['source_revision']='1d2ca42171570c713e78fc6a766ec5104b7f4724'
        save(state,out/'last.pth')
        save({'model':solver.ema.module.state_dict(),'last_epoch':epoch,'num_classes':12,'source':'EMA'},out/f'weights_epoch_{epoch+1:03d}.pth')
        if ap50_95>best:
            best=ap50_95;best_epoch=epoch+1;save(state,out/'best.pth')
        row={'epoch':epoch+1,'train':train,'validation':validation,'best_ap':best,'best_epoch':best_epoch,
             'seconds':time.monotonic()-start,'lr':[g['lr'] for g in solver.optimizer.param_groups]}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        (out/'status.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    (out/'COMPLETE').write_text(f'{epochs} epochs complete; local validation, no phase2 score\n')


if __name__=='__main__':main()
