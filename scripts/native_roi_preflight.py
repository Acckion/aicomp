"""Real CUDA preflight for original-pixel ROI, no val/test training."""
import copy,json,hashlib,os
from pathlib import Path
import torch
import native_roi
import extra_iou_metrics
from src.core import YAMLConfig
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'experiments/native_roi';out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(20260929);torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
cfg=YAMLConfig(str(ROOT/'configs/native_roi800.yml'));model=cfg.model.cuda();criterion=cfg.criterion.cuda()
parent=ROOT/'runs/ft_aug800/weights_epoch_020.pth';state=torch.load(parent,map_location='cpu',weights_only=False);weights=dict(state['ema']['module'] if 'ema' in state else state['model']);current=model.state_dict()
for key in ['decoder.anchors','decoder.valid_mask']:weights[key]=current[key]
missing=model.load_state_dict(weights,strict=False).missing_keys;assert all(k.startswith('native_roi.') for k in missing)
loader=cfg.train_dataloader;loader.set_epoch(0);packed,targets=next(iter(loader));packed=packed.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
model.eval()
with torch.no_grad():
    enabled=model(packed);model.roi_enabled=False;disabled=model(packed);model.roi_enabled=True
    diffs={k:float((enabled[k]-disabled[k]).abs().max()) for k in ['pred_logits','pred_boxes']};assert all(v==0 for v in diffs.values()),diffs
    ema=copy.deepcopy(model).eval();pred=ema(packed);assert torch.equal(enabled['pred_logits'],pred['pred_logits'])
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);model.train();gradients=[]
for i in range(2):
    optimizer.zero_grad(set_to_none=True)
    pred=model(packed,targets);loss=sum(criterion(pred,targets).values());assert torch.isfinite(loss)
    loss.backward();grads={name:float(p.grad.abs().sum()) for name,p in model.native_roi.named_parameters() if p.grad is not None};assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert grads['fusion.2.weight']>0
    if i:assert any(v>0 for k,v in grads.items() if k.startswith('encoder.'))
    gradients.append({'loss':float(loss),'roi_gradient_sums':grads});optimizer.step()
empty=[]
for t in targets:
    t=dict(t)
    for key in ['boxes','labels','area','iscrowd']:
        if key in t:t[key]=t[key][:0]
    empty.append(t)
optimizer.zero_grad(set_to_none=True);emptyloss=sum(criterion(model(packed,empty),empty).values());assert torch.isfinite(emptyloss);emptyloss.backward()
remote=ROOT/'checkpoints/gpu6_storage/native_roi';remote.mkdir(parents=True,exist_ok=True);check=remote/'preflight.pth';torch.save({'model':model.state_dict()},check)
clone=copy.deepcopy(model);clone.load_state_dict(torch.load(check,map_location='cpu',weights_only=True)['model'],strict=True);model.eval();clone.eval()
with torch.no_grad():assert torch.equal(model(packed)['pred_boxes'],clone(packed)['pred_boxes'])
check.unlink()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024**2),b''):h.update(c)
    return h.hexdigest()
train=loader.dataset;val=cfg.val_dataloader.dataset;assert not set(train.ids)&set(val.ids)
report={'stage':'passed','initial_exact_differences':diffs,'max_global_size':800,'native_batch_shape':list(packed.shape),'gradients':gradients,'empty_loss':float(emptyloss),'ema_deepcopy':True,'strict_checkpoint_reload':True,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'parent_sha256':sha(parent),'train_ann_sha256':sha(train.ann_file),'val_ann_sha256':sha(val.ann_file),'train_images':len(train),'val_images':len(val),'split_disjoint':True,'no_gt_crop_selection':True}
(out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
