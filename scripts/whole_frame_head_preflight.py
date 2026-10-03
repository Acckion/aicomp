"""Dense/empty GT updates, frozen-feature integrity and exact reload checks."""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import torch
import train_baseline
import taxonomy_pooling
import whole_frame_head
from src.core import YAMLConfig
from mechanism_runtime import freeze_bn_statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(20260929)
    cfg=YAMLConfig(str(ROOT/'configs'/f'{args.name}.yml'))
    cap=float(cfg.yaml_cfg['gpu_memory_limit_gib'])
    torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(0).total_memory)
    dataset=cfg.train_dataloader.dataset
    assert len(dataset)==1610 and len(cfg.val_dataloader.dataset)==390
    assert set(dataset.ids).isdisjoint(cfg.val_dataloader.dataset.ids)
    idx=dataset.ids.index(1537)
    packed,targets=cfg.train_dataloader.collate_fn([dataset[idx]])
    packed=packed.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
    model=cfg.model
    state=torch.load(ROOT/'runs/scene_obj365_pool800/weights_epoch_020.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    model.cuda();freeze_bn_statistics(model);model.train()
    ema=deepcopy(model).eval()
    criterion=cfg.criterion.cuda();optimizer=cfg.optimizer
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('backbone.','encoder.'))}
    # The two arms become identical when actual original pixels are replaced
    # by exactly the sham reconstruction. This checks packing and geometry.
    h,w=map(int,packed[0,6,0,:2].cpu().tolist());synthetic=packed.clone()
    synthetic[0,3:6,:h,:w]=torch.nn.functional.interpolate(packed[:,:3,:800,:800],size=(h,w),mode='bilinear',align_corners=False)[0]
    mode=model.mode;model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        model.mode='real';a=model(synthetic)
        model.mode='sham';b=model(synthetic)
        assert torch.equal(a['pred_logits'],b['pred_logits']) and torch.equal(a['pred_boxes'],b['pred_boxes'])
    del a,b,synthetic
    model.mode=mode;model.train();scaler=torch.amp.GradScaler('cuda',init_scale=1.)
    records=[]
    for step in range(3):
        t=deepcopy(targets)
        if step==2:
            for target in t:
                for key in ('boxes','labels','area','iscrowd'):
                    if key in target:target[key]=target[key][:0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=sum(criterion(model(packed,t),t,epoch=0).values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.backbone.parameters())
        assert all(p.grad is None for p in model.encoder.parameters())
        grads=[p.grad for p in model.decoder.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum()>0 for g in grads)
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1);scaler.step(optimizer);scaler.update()
        records.append({'step':step+1,'loss':float(loss.detach()),'empty_gt':step==2})
    assert all(torch.equal(value,model.state_dict()[key].cpu()) for key,value in before.items())
    model.eval();ema.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        expected=model(packed);other=ema(packed)
        assert torch.equal(expected['pred_boxes'],other['pred_boxes'])
        assert torch.equal(expected['pred_logits'],other['pred_logits'])
    del other
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    model.load_state_dict(torch.load(stream,map_location='cuda',weights_only=True),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        other=model(packed)
        assert torch.equal(expected['pred_boxes'],other['pred_boxes'])
        assert torch.equal(expected['pred_logits'],other['pred_logits'])
    report={'stage':'passed','name':args.name,'mode':mode,'train_gt':len(targets[0]['labels']),
        'geometry':[model.height,model.width],'records':records,'ema_resident':True,
        'frozen_visual_state_exact':True,'strict_reload_exact':True,'sham_reconstruction_identity_exact':True,
        'memory_cap_gib':cap,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'limitations':'One dense training image and three actual updates; no AP or online claim.'}
    out=ROOT/'experiments/whole_frame_head'/args.name;out.mkdir(parents=True,exist_ok=True)
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
