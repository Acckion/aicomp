"""Recover known COCO classifier rows; no external images or extra inference."""
import json
import torch
import train_baseline as baseline
from src.data.dataset.coco_dataset import mscoco_category2name,mscoco_category2label

# Five exact category matches; chair is a declared partial prior for seat.
CLASS_NAMES={0:'person',1:'boat',3:'chair',5:'bicycle',6:'car',7:'sports ball'}
BY_NAME={name:mscoco_category2label[c] for c,name in mscoco_category2name.items()}
MAPPING={target:BY_NAME[name] for target,name in CLASS_NAMES.items()}
_previous=baseline.BaselineSolver.load_tuning_state


def load(solver,path):
    if not solver.cfg.yaml_cfg.get('semantic_row_init',False):return _previous(solver,path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    weights=dict(state['ema']['module'] if 'ema' in state else state['model'])
    current=solver.model.state_dict();changed=[]
    for key in ('decoder.anchors','decoder.valid_mask'):
        if key in current:weights[key]=current[key]
    for key,target in current.items():
        if 'score_head' in key and key in weights and weights[key].shape!=target.shape:
            source=weights[key];assert source.shape[0]==80 and target.shape[0]==12 and source.shape[1:]==target.shape[1:],(key,source.shape,target.shape)
            value=target.clone()
            for dst,src in MAPPING.items():value[dst].copy_(source[src])
            unmapped=[i for i in range(12) if i not in MAPPING]
            assert torch.equal(value[unmapped],target[unmapped])
            assert all(torch.equal(value[dst],source[src].to(value.dtype)) for dst,src in MAPPING.items())
            weights[key]=value;changed.append(key)
        elif key=='decoder.denoising_class_embed.weight':
            source=weights[key];assert source.shape[0]==81 and target.shape[0]==13 and source.shape[1:]==target.shape[1:]
            value=target.clone()
            for dst,src in MAPPING.items():value[dst].copy_(source[src])
            value[12].copy_(source[80]);weights[key]=value;changed.append(key)
    assert 'decoder.enc_score_head.weight' in changed and 'decoder.enc_score_head.bias' in changed
    assert any('dec_score_head' in k for k in changed)
    matched,info=solver._matched_state(current,weights)
    assert not info['missed'] and not info['unmatched'],info
    solver.model.load_state_dict(matched,strict=True)
    solver.semantic_initialization={'mapping':MAPPING,'names':CLASS_NAMES,'changed_tensors':changed,
        'seat_prior':'chair only, not exact equivalence to all seat objects','unknown_classes':'original initialization, no invented COCO match'}
    print('SEMANTIC_ROW_INITIALIZATION '+json.dumps(solver.semantic_initialization),flush=True)


baseline.BaselineSolver.load_tuning_state=load
