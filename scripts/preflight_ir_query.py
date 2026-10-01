"""Real paired-data, initialization, alignment-mask and CUDA-gradient checks."""
import argparse
import json
from pathlib import Path
import random
import torch
import ir_query_alignment
from src.core import YAMLConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/ir_query_alignment'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    random.seed(20261002);torch.manual_seed(20261002)
    cfg=YAMLConfig(str(ROOT/'configs/ir_query800.yml'))
    dataset=cfg.train_dataloader.dataset
    validation=cfg.val_dataloader.dataset
    assert len(dataset)==1600 and len(validation)==400
    assert set(dataset.ids).isdisjoint(validation.ids)
    rows=[]
    for index in range(0,1600,25):
        sample,target=dataset[index]
        assert sample.shape==(7,800,800)
        assert len(target['labels'])==len(target['boxes'])
        assert torch.isfinite(sample).all() and torch.isfinite(target['boxes']).all()
        assert bool(((target['boxes']>=0)&(target['boxes']<=1)).all())
        fractions=[]
        for cx,cy,w,h in target['boxes']:
            x0=max(0,int((cx-w/2)*800));x1=min(800,int((cx+w/2)*800)+1)
            y0=max(0,int((cy-h/2)*800));y1=min(800,int((cy+h/2)*800)+1)
            fractions.append(float(sample[6,y0:y1,x0:x1].mean()))
        rows.append({'image_id':dataset.ids[index],'ir_valid_fraction':float(sample[6].mean()),
                     'objects':len(fractions),'gt_under50pct_valid_ir':sum(f<.5 for f in fractions),
                     'gt_ir_valid_fractions':fractions})
    report={'training_images_sampled':len(rows),'train_validation_disjoint':True,
            'validity_definition':'near-zero components connected to image border at 800 pixels; visibility proxy only',
            'rows':rows,'objects':sum(r['objects']for r in rows),
            'gt_under50pct_valid_ir':sum(r['gt_under50pct_valid_ir']for r in rows)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'train_validity_audit.json').write_text(json.dumps(report,indent=2))
    print('TRAIN_PAIR_AUDIT',json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)
    if not args.gpu:return
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=cfg.model.cuda();state=torch.load(ROOT/'runs/ft_aug800/weights_epoch_020.pth',map_location='cpu',weights_only=False)['model']
    current=model.state_dict();state={**state,'decoder.anchors':current['decoder.anchors'],'decoder.valid_mask':current['decoder.valid_mask']}
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and all(k.startswith('ir_query.')for k in result.missing_keys)
    sample,target=dataset[10];samples=sample.unsqueeze(0).cuda()
    target={k:v.cuda()if isinstance(v,torch.Tensor)else v for k,v in target.items()}
    model.eval()
    with torch.no_grad():
        model.ir_enabled=False;rgb=model(samples)
        model.ir_enabled=True;paired=model(samples)
        assert torch.equal(rgb['pred_boxes'],paired['pred_boxes'])
        assert torch.equal(rgb['pred_logits'],paired['pred_logits'])
        from copy import deepcopy
        ema=deepcopy(model)
        assert ema.decoder.decoder._forward_pre_hooks[next(iter(ema.decoder.decoder._forward_pre_hooks))].__self__ is ema
        del ema,rgb,paired
    criterion=cfg.criterion.cuda();optimizer=cfg.optimizer
    samples=torch.nn.functional.interpolate(samples,size=(992,992))
    from mechanism_runtime import freeze_bn_statistics
    freeze_bn_statistics(model);model.train()
    projection_grad=[];offset_grad=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            outputs=model(samples,[target]);losses=criterion(outputs,[target],epoch=0)
            loss=sum(losses.values())
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:assert torch.isfinite(parameter.grad).all()
        projection_grad.append(float(model.ir_query.output_projection.weight.grad.norm()))
        offset_grad.append(float(model.ir_query.offset.weight.grad.norm()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1);optimizer.step()
    assert projection_grad[0]>0 and offset_grad[1]>0
    model.eval()
    with torch.no_grad():
        model.ir_enabled=False;rgb=model(sample.unsqueeze(0).cuda())
        model.ir_enabled=True;model.ir_mode='zero';zero=model(sample.unsqueeze(0).cuda())
        assert torch.equal(rgb['pred_boxes'],zero['pred_boxes']) and torch.equal(rgb['pred_logits'],zero['pred_logits'])
    result={'stage':'passed','initial_native_predictions_exact':True,'ema_hook_owner_correct':True,
            'zero_ir_predictions_exact_after_update':True,'projection_gradient':projection_grad,
            'offset_gradient':offset_grad,'peak_allocated_mib_batch1_992':torch.cuda.max_memory_allocated()/1024**2}
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2));print('IR_QUERY_PREFLIGHT_PASSED',json.dumps(result),flush=True)

if __name__=='__main__':main()
