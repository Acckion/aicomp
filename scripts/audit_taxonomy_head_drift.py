"""CPU-only source-classifier drift audit; no inference or model updates."""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
HEADS=['decoder.enc_score_head']+[f'decoder.dec_score_head.{i}' for i in range(6)]


def groups():
    module=ast.parse((ROOT/'scripts/taxonomy_pooling.py').read_text())
    return next(ast.literal_eval(n.value) for n in module.body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPS' for t in n.targets))


def load_heads(path,pooled):
    state=torch.load(path,map_location='cpu',weights_only=False)
    use_ema='ema' in state
    weights=state['ema']['module'] if use_ema else state['model']
    out={}
    for prefix in HEADS:
        source=prefix+'.source' if pooled else prefix
        w=weights[source+'.weight'].float().clone();b=weights[source+'.bias'].float().clone()
        assert w.shape==(366,256) and b.shape==(366,)
        out[prefix]={'weight':w,'bias':b}
        if pooled:
            out[prefix]['residual_weight']=weights[prefix+'.residual.weight'].float().clone()
            out[prefix]['residual_bias']=weights[prefix+'.residual.bias'].float().clone()
    meta={'epoch':state.get('last_epoch',-1)+1,'weights':'EMA' if use_ema else 'model',
          'path':str(path),'bytes':path.stat().st_size}
    del state,weights;gc.collect()
    return out,meta


def compare(reference,current,ids):
    a=reference['weight'][ids];b=current['weight'][ids]
    return {'mean_row_cosine':float(F.cosine_similarity(a,b,dim=1).mean()),
            'relative_weight_l2':float((b-a).norm()/a.norm().clamp_min(1e-12)),
            'mean_absolute_bias_change':float((current['bias'][ids]-reference['bias'][ids]).abs().mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'experiments/taxonomy_head_drift'));a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'report.json').exists(),'Use a new output to preserve prior audit'
    source=ROOT/'checkpoints/dfine_x_obj365.pth'
    first=ROOT/'runs/scene_obj365_pool800/last_before_vectorization.pth'
    current=out/'current_full.pth'
    if not current.exists():os.link(ROOT/'runs/scene_obj365_pool800/last.pth',current)
    torch.set_num_threads(2)
    public,public_meta=load_heads(source,False)
    one,one_meta=load_heads(first,True)
    latest,latest_meta=load_heads(current,True)
    taxonomy=groups();mapped=sorted(set(i for ids in taxonomy.values() for i in ids));unused=sorted(set(range(366))-set(mapped))
    known=sorted(taxonomy)
    result={}
    for prefix in HEADS:
        result[prefix]={'epoch1_mapped':compare(public[prefix],one[prefix],mapped),
            'latest_mapped':compare(public[prefix],latest[prefix],mapped),
            'latest_unused':compare(public[prefix],latest[prefix],unused),
            'latest_groups':{str(c):compare(public[prefix],latest[prefix],ids) for c,ids in taxonomy.items()},
            'epoch1_known_residual_weight_rms':float(one[prefix]['residual_weight'][known].square().mean().sqrt()),
            'latest_known_residual_weight_rms':float(latest[prefix]['residual_weight'][known].square().mean().sqrt())}
    report={'created_at':time.time(),'reference':public_meta,'epoch1':one_meta,'latest':latest_meta,
        'heads':result,'uses_images':False,'uses_validation_labels':False,'updates':0,
        'limitations':'Parameter drift is descriptive. It cannot prove semantic forgetting, identify AP drop causes, or establish that freezing improves detection. Features also adapt; unchanged unused rows are only a reference.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'epoch':latest_meta['epoch'],'encoder':result[HEADS[0]],'last_decoder':result[HEADS[-1]]}),flush=True)


if __name__=='__main__':main()
