"""True max-scale training, empty GT, EMA ownership and zero-IR checks."""
import argparse,json,random,io
from copy import deepcopy
from pathlib import Path
import torch
import ir_content_alignment
from src.core import YAMLConfig
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/full2000_ir'
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.parse_args()
    torch.manual_seed(20261002);random.seed(20261002)
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=YAMLConfig(str(ROOT/'configs/full2000_ir_content800.yml'));dataset=cfg.train_dataloader.dataset
    assert len(dataset)==2000
    assert len(set(dataset.ids))==2000
    assert cfg.yaml_cfg['validate'] is False
    model=cfg.model.cuda()
    state=torch.load(ROOT/'runs/ft2000_aug800/weights_epoch_020.pth',map_location='cpu',weights_only=False)['model']
    current=model.state_dict()
    for k in ('decoder.anchors','decoder.valid_mask'):state[k]=current[k]
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and all(k.startswith(('ir_backbone.','ir_encoder.','ir_samplers.')) for k in result.missing_keys)
    del state,current
    samples=[];targets=[]
    for i in [10,11]:
        sample,target=dataset[i];samples.append(sample)
        targets.append({k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in target.items()})
    x=torch.stack(samples).cuda();model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        model.ir_enabled=False;a=model(x);model.ir_enabled=True;b=model(x)
        assert torch.equal(a['pred_boxes'],b['pred_boxes']) and torch.equal(a['pred_logits'],b['pred_logits'])
        del a,b
        ema=deepcopy(model)
        for layer in ema.decoder.decoder.layers[-3:]:
            hook=list(layer._forward_pre_hooks.values())[-1];assert hook.__self__ is ema
        # Keep EMA resident during the peak training check.
    criterion=cfg.criterion.cuda();optimizer=cfg.optimizer
    from mechanism_runtime import freeze_bn_statistics
    freeze_bn_statistics(model);model.train()
    x=torch.nn.functional.interpolate(x,size=(992,992));scaler=torch.amp.GradScaler('cuda',init_scale=1)
    grads=[]
    for step in range(3):
        t=deepcopy(targets)
        if step==2:
            for target in t:
                target['boxes']=target['boxes'][:0];target['labels']=target['labels'][:0]
                if 'area' in target:target['area']=target['area'][:0]
                if 'iscrowd' in target:target['iscrowd']=target['iscrowd'][:0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            losses=criterion(model(x,t),t,epoch=0);loss=sum(losses.values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()];assert not bad,bad
        record={n:float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith('ir_samplers.') and p.grad is not None}
        grads.append(record)
        assert all(p.grad is None for p in model.ir_backbone.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1);scaler.step(optimizer);scaler.update()
        print('IR_CONTENT_PREFLIGHT_STEP',step,float(loss),flush=True)
    assert any(v>0 for k,v in grads[0].items() if '.output.' in k)
    assert any(v>0 for k,v in grads[1].items() if '.shift.' in k)
    model.eval();model.ir_mode='zero'
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        test=torch.stack(samples).cuda();model.ir_enabled=False;a=model(test);model.ir_enabled=True;b=model(test)
        assert torch.equal(a['pred_boxes'],b['pred_boxes']) and torch.equal(a['pred_logits'],b['pred_logits'])
        del a,b
        buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);model.load_state_dict(torch.load(buf,map_location='cuda',weights_only=True),strict=True)
        del buf
    report={'stage':'passed','strict_teacher_loaded':model.ir_teacher_source,'initial_native_exact':True,'zero_ir_after_update_exact':True,'empty_gt_finite':True,'ema_hook_owner_correct':True,'strict_reload_passed':True,'max_scale':992,'batch_size':2,'ema_resident':True,'training_images':2000,'independent_validation':False,'optimizer_updates':3,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'gradients':grads}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'preflight.json').write_text(json.dumps(report,indent=2));print('IR_CONTENT_PREFLIGHT_PASSED',json.dumps(report),flush=True)
if __name__=='__main__':main()
