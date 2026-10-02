"""Maximum-scale throughput/memory probe, no training checkpoint artifacts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import taxonomy_pooling
import train_baseline as baseline
from src.core import YAMLConfig
from mechanism_runtime import install_training_controls

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,choices=[1,2],required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    out=ROOT/'experiments/scene_batch_probe';out.mkdir(parents=True,exist_ok=True)
    cfg=YAMLConfig(str(ROOT/'configs/scene_obj365_pool800.yml'),output_dir=str(out))
    cfg.tuning=str(ROOT/'checkpoints/dfine_x_obj365.pth')
    install_training_controls(baseline)
    solver=baseline.BaselineSolver(cfg);solver._setup()
    loader=cfg.train_dataloader
    # Same two augmented train samples and effective batch8 in both processes.
    iterator=iter(loader);samples=[next(iterator),next(iterator)]
    samples=[(F.interpolate(x.cuda(),size=(992,992),mode='bilinear',align_corners=False),
              [{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]) for x,targets in samples]
    optimizer=cfg.optimizer;model=solver.model.train();scaler=torch.amp.GradScaler('cuda',init_scale=1.)
    updates=[];hook=optimizer.register_step_post_hook(lambda *args:updates.append(True));records=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        for micro in range(8//a.batch_size):
            selected=[samples[(micro*a.batch_size+j)%2] for j in range(a.batch_size)]
            x=torch.cat([s[0] for s in selected]);targets=deepcopy([t for s in selected for t in s[1]])
            if step==2 and micro==8//a.batch_size-1:
                for t in targets:
                    for k in ('boxes','labels','area','iscrowd'):
                        if k in t:t[k]=t[k][:0]
            with torch.autocast('cuda',dtype=torch.float16):
                loss=sum(solver.criterion(model(x,targets),targets,epoch=0).values())/(8//a.batch_size)
            assert torch.isfinite(loss);scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
        scaler.step(optimizer);scaler.update();solver.ema.update(model);torch.cuda.synchronize()
        record={'update':step+1,'seconds_for_effective_batch8':time.perf_counter()-start,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'last_micro_loss':float(loss)}
        records.append(record);print(json.dumps(record),flush=True)
    assert len(updates)==3;hook.remove()
    result={'stage':'passed','batch_size':a.batch_size,'accumulation':8//a.batch_size,'maximum_scale':992,
            'ema_resident':True,'actual_finite_updates':len(updates),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'images_per_second_last_two_updates':16/sum(r['seconds_for_effective_batch8'] for r in records[-2:]),'records':records,
            'limitations':'Fixed train examples, shared GPU speed snapshot; not AP or full-run stability. No existing training interrupted.'}
    (out/f'batch{a.batch_size}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
