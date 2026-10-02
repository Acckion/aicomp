"""Dense992 batch1 with resident EMA; three real finite updates and reload."""
import copy,json,torch
from pathlib import Path
import vf_tokens_adapter,extra_iou_metrics
from mechanism_runtime import freeze_bn_statistics
from src.core import YAMLConfig
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/vf_tokens';OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(2);torch.manual_seed(20261002);torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
cfg=YAMLConfig(str(ROOT/'configs/vf_tokens800.yml'));model=cfg.model.cuda();criterion=cfg.criterion.cuda();freeze_bn_statistics(model)
parent=ROOT/'runs/ft_aug800/weights_epoch_020.pth';state=torch.load(parent,map_location='cpu',weights_only=False);weights=dict(state['ema']['module'] if 'ema' in state else state['model']);del state
current=model.state_dict()
for k in ('decoder.anchors','decoder.valid_mask'):weights[k]=current[k]
missing=model.load_state_dict(weights,strict=False).missing_keys;assert all(k.startswith('vf_adapter.') for k in missing)
del weights,current
loader=cfg.train_dataloader;dataset=loader.dataset
idx=max(range(len(dataset)),key=lambda i:len(dataset.coco.getAnnIds(imgIds=[dataset.ids[i]])))
x,targets=loader.collate_fn([dataset[idx]]);x=x.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
model.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    a=model(x);model.vf_enabled=False;b=model(x);model.vf_enabled=True
    diffs={k:float((a[k]-b[k]).abs().max()) for k in ('pred_logits','pred_boxes')};assert all(v==0 for v in diffs.values()),diffs
    ema=copy.deepcopy(model).eval();assert next(iter(ema.encoder.input_proj[0]._forward_hooks.values())).__self__ is ema
    assert torch.equal(a['pred_logits'],ema(x)['pred_logits'])
del a,b;torch.cuda.empty_cache()
optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.startswith('vf_adapter.')],'lr':1e-5},{'params':model.vf_adapter.parameters(),'lr':3e-4}]);scaler=torch.amp.GradScaler('cuda',init_scale=1.);records=[]
eval_x=x; x=torch.nn.functional.interpolate(x,size=(992,992),mode='bilinear',align_corners=False)
for step in range(3):
    active=targets
    if step==2:
        active=[]
        for t in targets:
            t=dict(t)
            for k in ('boxes','labels','area','iscrowd'):
                if k in t:t[k]=t[k][:0]
            active.append(t)
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=sum(criterion(model(x,active),active).values())
    assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    grads={n:float(p.grad.abs().sum()) for n,p in model.vf_adapter.named_parameters() if p.grad is not None};assert grads['output.weight']>0
    if step==1:assert grads['project.weight']>0
    assert all(p.grad is None for p in vf_tokens_adapter.teacher(x.device).parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),.1);scaler.step(optimizer);scaler.update()
    records.append({'step':step+1,'empty_gt':step==2,'loss':float(loss),'adapter_gradients':grads});print(json.dumps(records[-1]),flush=True)
model.eval();x=eval_x
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    normal=model(x);model.vf_mode='zero';zero=model(x);model.vf_enabled=False;disabled=model(x);model.vf_enabled=True;model.vf_mode='real'
    assert torch.equal(zero['pred_logits'],disabled['pred_logits']);delta=float((normal['pred_logits']-zero['pred_logits']).abs().max());assert delta>0
check=ROOT/'checkpoints/gpu6_storage/vf_tokens/preflight.pth';torch.save({'model':model.state_dict()},check);ema.load_state_dict(torch.load(check,map_location='cpu',weights_only=True)['model'],strict=True)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(normal['pred_logits'],ema(x)['pred_logits'])
check.unlink();val=cfg.val_dataloader.dataset;assert len(dataset)==1600 and len(val)==400 and not set(dataset.ids)&set(val.ids)
report={'stage':'passed','actual_finite_optimizer_steps':3,'max_input_size':[992,992],'batch_size':1,'ema_resident':True,'initial_exact_differences':diffs,'records':records,'zero_feature_equals_disabled':True,'real_feature_effect_after_updates':delta,'teacher_frozen':True,'ema_bound_owner':True,'strict_reload':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'train1600_val400_disjoint':True}
(OUT/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
