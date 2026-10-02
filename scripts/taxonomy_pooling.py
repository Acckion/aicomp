"""Transfer fine-grained public classifier knowledge into 12 supervised classes.

One detector and one box output. No external images, separate teacher or voting.
Raw Objects365 v2 IDs are score rows because upstream does not remap labels.
"""
import json
import torch
from torch import nn
import train_baseline as baseline
from src.zoo.dfine.dfine_decoder import DFINETransformer

GROUPS = {
    0: [1],
    1: [22, 73, 212],
    2: [57, 79, 93, 97, 100, 104, 137, 140, 145, 165, 179, 181,
        222, 241, 242, 258, 263, 274, 296, 301, 308, 319, 320, 321,
        322, 324, 325, 327, 331, 342, 343, 345],
    3: [3, 25, 48, 51],
    4: [90, 128, 268, 306],
    5: [47],
    6: [6, 35, 50, 88, 127],
    7: [119, 157, 166, 178, 207, 240, 248],
    8: [7, 12, 41],
    9: [45],
    11: [184],
}


class TaxonomyScoreHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.out_features = 12
        self.source = nn.Linear(original.in_features, 366)
        self.residual = nn.Linear(original.in_features, 12)
        with torch.no_grad():
            self.residual.weight.copy_(original.weight)
            self.residual.bias.copy_(original.bias)
            for target in GROUPS:
                self.residual.weight[target].zero_()
                self.residual.bias[target].zero_()
        for target, ids in GROUPS.items():
            self.register_buffer('rows_'+str(target), torch.tensor(ids, dtype=torch.long))

    def forward(self, x):
        prior = self.source(x)
        # Max preserves the highest matching subtype's pretrained logit.
        # Averaging animal or light rows can cancel mutually distinct evidence.
        columns = [prior.index_select(-1, getattr(self, 'rows_'+str(c))).amax(-1)
                   if c in GROUPS else torch.zeros_like(prior[..., 0]) for c in range(12)]
        return torch.stack(columns, -1) + self.residual(x)


_init = DFINETransformer.__init__
def initialize(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    assert self.num_classes == 12
    # Creating the added modules must not perturb augmentation RNG.
    with torch.random.fork_rng(devices=[]):
        self.enc_score_head = TaxonomyScoreHead(self.enc_score_head)
        self.dec_score_head = nn.ModuleList(TaxonomyScoreHead(h) for h in self.dec_score_head)
DFINETransformer.__init__ = initialize


def load(solver, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    source = dict(state['ema']['module'] if 'ema' in state else state['model'])
    current = solver.model.state_dict()
    if 'decoder.enc_score_head.source.weight' in source:
        for key in ('decoder.anchors', 'decoder.valid_mask'):
            if key in current:
                source[key] = current[key]
        for key in current:
            if '.rows_' in key:
                assert torch.equal(source[key], current[key]), ('Changed taxonomy', key)
        solver.model.load_state_dict(source, strict=True)
        solver.semantic_initialization = {'source':'trained taxonomy-pooling detector', 'groups':GROUPS}
        return
    copied = dict(current)
    head_names = ['decoder.enc_score_head'] + ['decoder.dec_score_head.'+str(i) for i in range(6)]
    changed = set()
    for name in head_names:
        for suffix in ('weight', 'bias'):
            original_key = name+'.'+suffix
            target_key = name+'.source.'+suffix
            assert source[original_key].shape == current[target_key].shape
            assert source[original_key].shape[0] == 366
            copied[target_key] = source[original_key]
            changed.add(original_key)
    dn = 'decoder.denoising_class_embed.weight'
    assert source[dn].shape == (367, current[dn].shape[1])
    copied[dn] = current[dn].clone()
    for target, rows in GROUPS.items():
        copied[dn][target] = source[dn][rows].mean(0)
    copied[dn][12] = source[dn][366]
    changed.add(dn)
    unexpected = []
    for key, tensor in source.items():
        if key in changed or key in ('decoder.anchors', 'decoder.valid_mask'):
            continue
        if key not in current or tensor.shape != current[key].shape:
            unexpected.append(key)
        else:
            copied[key] = tensor
    assert not unexpected, unexpected
    expected_new = [k for k in current if k not in source]
    assert all(any(k.startswith(n+'.') for n in head_names) for k in expected_new), expected_new
    solver.model.load_state_dict(copied, strict=True)
    solver.semantic_initialization = {'source_classes':366, 'groups':GROUPS,
        'method':'trainable source subtype max logits plus trainable target residual',
        'unknown_uav':'original target initialization; no airplane-as-UAV assumption',
        'partial_priors':'Groups are semantic initialization only, fine-tuned using official train GT; not exact category equivalence.'}
    print('TAXONOMY_INITIALIZATION '+json.dumps(solver.semantic_initialization), flush=True)
    # Assert exact initial pooling and unknown-class identity without image GT.
    head = solver.model.decoder.enc_score_head
    x = head.source.weight.new_ones((2, head.source.in_features))
    with torch.no_grad():
        logits, original = head(x), head.source(x)
        for c, rows in GROUPS.items():
            assert torch.equal(logits[:,c], original[:,rows].amax(-1))
        assert torch.equal(logits[:,10], head.residual(x)[:,10])


baseline.BaselineSolver.load_tuning_state = load
