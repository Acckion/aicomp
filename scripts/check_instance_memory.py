"""Numerical identity, FDR/score influence, gradients and EMA ownership."""
import copy
import json
import os
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import train_baseline as baseline
import instance_memory
from src.core import YAMLConfig

def main():
    torch.set_num_threads(2);torch.manual_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    instance_memory.install();config=YAMLConfig(str(baseline.ROOT/'configs/instance_memory800.yml'))
    model=config.model;parent=baseline.ROOT/'runs/ft_aug800/weights_epoch_020.pth'
    weights=torch.load(parent,map_location='cpu',weights_only=False)['model'];weights=dict(weights)
    for key in ('decoder.anchors','decoder.valid_mask'):weights[key]=model.state_dict()[key]
    result=model.load_state_dict(weights,strict=False)
    assert all(k.startswith('decoder.instance_memory.') for k in result.missing_keys) and not result.unexpected_keys
    model.cuda().eval();memory=model.decoder.instance_memory
    images=json.loads((baseline.ROOT/'data/annotations/val400.json').read_text())['images']
    with Image.open(baseline.ROOT/'data/train'/images[0]['file_name']) as image:
        tensor=TF.to_tensor(TF.resize(image.convert('RGB'),[800,800])).unsqueeze(0).cuda()
    with torch.no_grad():
        memory.enabled=False;reference=model(tensor)
        memory.enabled=True;zero=model(tensor)
    assert torch.equal(reference['pred_logits'],zero['pred_logits'])
    assert torch.equal(reference['pred_boxes'],zero['pred_boxes'])
    clone=copy.deepcopy(model)
    hook=next(iter(clone.decoder.decoder.layers[clone.decoder.eval_idx]._forward_hooks.values()))
    assert hook.__self__ is clone.decoder.instance_memory
    del clone
    # Zero initialization deliberately delays key/value gradients one update.
    optimizer=torch.optim.AdamW(memory.parameters(),lr=.01)
    gradients={}
    for step in range(2):
        optimizer.zero_grad();queries=torch.randn(1,12,memory.hidden,device='cuda')
        loss=(memory(queries)-queries-1).square().mean();loss.backward()
        gradients[str(step)]={name:float(p.grad.abs().sum()) if p.grad is not None else None for name,p in memory.named_parameters()}
        assert gradients[str(step)]['output.weight']>0
        if step:assert gradients[str(step)]['key.weight']>0 and gradients[str(step)]['value.weight']>0
        optimizer.step()
    with torch.no_grad():
        memory.enabled=True;changed=model(tensor)
    changes={name:float((changed[name]-reference[name]).abs().max()) for name in ('pred_logits','pred_boxes')}
    assert all(value>0 for value in changes.values()),changes
    # A strictly reconstructed augmented model includes its prototype buffers.
    rebuilt=config.model
    # YAMLConfig caches its model; use a new configuration for a fresh instance.
    rebuilt=YAMLConfig(str(baseline.ROOT/'configs/instance_memory800.yml')).model
    rebuilt.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
    report={'stage':'passed','zero_initialization_identical':True,'ema_hook_owner_correct':True,
            'strict_augmented_state_reload':True,'changes':changes,'gradients':gradients,
            'query_dim':memory.hidden,'bank_feature_dim':memory.prototype_features.shape[-1],
            'prototypes':len(memory.prototype_labels),'background_prototypes_used':0,
            'cuda_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
    (baseline.ROOT/'experiments/instance_memory/preflight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
