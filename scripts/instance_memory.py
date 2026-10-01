"""Train-instance retrieval inside D-FINE, shared by scoring and FDR regression.

This is an independent experiment inspired by instance caching, not a faithful
X-Few reproduction. A frozen train-only bank provides appearance, 2x2 spatial
structure and size statistics. No image identity or coordinates are queried.
"""
from pathlib import Path
import functools
import hashlib
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'checkpoints/gpu6_storage/instance_memory/memory.npz'
ENABLED=True

def compact_bank(bank):
    """Read the large regional archive once, persist only inference prototypes."""
    bank=Path(bank);cache=bank.with_name('query_prototypes_v1.pth')
    identity=hashlib.sha256(bank.with_name('identity.json').read_bytes()).hexdigest()
    if cache.exists():
        compact=torch.load(cache,map_location='cpu',weights_only=True)
        if compact['identity_sha256']==identity:return compact
    with np.load(bank) as data:
        features=torch.from_numpy(data['features']).float()
        spatial=torch.from_numpy(data['spatial_features']).float()
        labels=torch.from_numpy(data['labels']).long()
        prototypes=torch.from_numpy(data['prototypes']).float()
        proto_labels=torch.from_numpy(data['prototype_labels']).long()
        geometry=torch.from_numpy(data['box_wh']).float()
    keep=proto_labels<12;prototypes=prototypes[keep];proto_labels=proto_labels[keep]
    structures=[];sizes=[]
    for i,label in enumerate(proto_labels):
        members=(labels==label).nonzero().flatten()
        class_protos=(proto_labels==label).nonzero().flatten()
        assignment=(features[members]@prototypes[class_protos].T).argmax(-1)
        local=int((class_protos==i).nonzero().flatten()[0]);selected=members[assignment==local]
        if not len(selected):selected=members
        structures.append(F.normalize(spatial[selected].mean(0),dim=0))
        sizes.append(geometry[selected].clamp_min(1e-4).log().mean(0))
    compact={'identity_sha256':identity,'features':prototypes,'structure':torch.stack(structures),
             'geometry':torch.stack(sizes),'labels':proto_labels}
    temporary=cache.with_suffix(f'.{os.getpid()}.tmp');torch.save(compact,temporary);temporary.replace(cache)
    return compact

class InstanceMemory(nn.Module):
    def __init__(self, hidden, bank=BANK):
        super().__init__()
        compact=compact_bank(bank)
        prototypes=compact['features'];proto_labels=compact['labels']
        # Candidate backgrounds are not verified negatives: exclude them from
        # this primary experiment instead of treating them as a 13th class.
        self.register_buffer('prototype_features',prototypes)
        self.register_buffer('prototype_structure',compact['structure'])
        self.register_buffer('prototype_geometry',compact['geometry'])
        self.register_buffer('prototype_labels',proto_labels)
        self.query_norm=nn.LayerNorm(hidden)
        self.key=nn.Linear(prototypes.shape[-1],hidden,bias=False)
        self.value=nn.Linear(compact['structure'].shape[-1],hidden,bias=False)
        self.geometry=nn.Linear(2,hidden,bias=False)
        self.class_embedding=nn.Embedding(12,hidden)
        self.output=nn.Linear(hidden,hidden)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.enabled=ENABLED
        self.hidden=hidden

    def forward(self,queries):
        if not self.enabled:return queries
        query=F.normalize(self.query_norm(queries),dim=-1)
        keys=F.normalize(self.key(self.prototype_features),dim=-1)
        attention=(query@keys.T/.1).softmax(-1)
        values=(self.value(self.prototype_structure)+self.geometry(self.prototype_geometry)*.1
                +self.class_embedding(self.prototype_labels)*.1)
        return queries+self.output(attention@values)*.1

    def forward_hook(self,module,args,output):
        if output.shape[-1]!=self.hidden:
            raise RuntimeError(f'Instance memory query width mismatch: {output.shape[-1]} vs {self.hidden}')
        return self(output)

def install(enabled=True,bank=BANK):
    """Keep upstream parameter names, add one registered memory to eval layer."""
    global ENABLED,BANK
    ENABLED=enabled;BANK=Path(bank)
    from src.zoo.dfine.dfine_decoder import DFINETransformer
    if getattr(DFINETransformer.__init__,'_instance_memory',False):return
    original=DFINETransformer.__init__
    @functools.wraps(original)
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        hidden=self.dec_score_head[self.eval_idx].in_features
        self.instance_memory=InstanceMemory(hidden,BANK)
        # The modified query feeds both the score head and corner-distribution
        # regression, and propagates to later training-only decoder layers.
        self.decoder.layers[self.eval_idx].register_forward_hook(self.instance_memory.forward_hook)
    init._instance_memory=True;DFINETransformer.__init__=init

def install_initialization(baseline):
    def load(solver,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        weights=dict(state['ema']['module'] if 'ema' in state else state['model'])
        current=solver.model.state_dict()
        for key in ('decoder.anchors','decoder.valid_mask'):weights[key]=current[key]
        missing=set(current)-set(weights);unexpected=set(weights)-set(current)
        assert not unexpected,unexpected
        assert all(k.startswith('decoder.instance_memory.') for k in missing),missing
        mismatch=[k for k in weights if weights[k].shape!=current[k].shape]
        assert not mismatch,mismatch
        result=solver.model.load_state_dict(weights,strict=False)
        assert set(result.missing_keys)==missing and not result.unexpected_keys
        print(f'INSTANCE_MEMORY_INIT parent={len(weights)} new={sorted(missing)}',flush=True)
    baseline.BaselineSolver.load_tuning_state=load
