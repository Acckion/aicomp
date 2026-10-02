"""Train-only fine-subtype denoising hints with unchanged coarse detection labels.

Only class-compatible source proposals overlapping train GT (IoU>=.5 and
source sigmoid>=.3) supply a .25 frozen source-embedding residual. Label
noise, padding, unknown UAV, query budgets and native inference stay intact.
This is a hypothesis about lost subtype information, not a demonstrated gain.
"""
import torch
from torch import nn
from torchvision.ops import box_convert, box_iou
import taxonomy_pooling as pooling
import train_baseline as baseline
from src.zoo.dfine import dfine_decoder as decoder_module

Decoder = decoder_module.DFINETransformer
_init = Decoder.__init__
_forward = Decoder.forward
_encoder = Decoder._get_encoder_input
_dn = decoder_module.get_contrastive_denoising_training_group
_load = baseline.BaselineSolver.load_tuning_state
STRENGTH = .25


class ConditionalEmbedding:
    def __init__(self, base, labels, delta, supported):
        self.base, self.labels, self.delta, self.supported = base, labels, delta, supported

    def __call__(self, noisy_labels):
        output = self.base(noisy_labels)
        maximum = self.labels.shape[1]
        assert maximum > 0 and noisy_labels.shape[1] % maximum == 0
        repeat = noisy_labels.shape[1] // maximum
        expected = self.labels.repeat(1, repeat)
        keep = self.supported.repeat(1, repeat) & (noisy_labels == expected)
        delta = self.delta.repeat(1, repeat, 1).to(output.dtype)
        return output + STRENGTH * delta * keep[..., None]


def initialize(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    assert self.num_classes == 12
    self.enc_score_head.register_buffer('fine_dn_bank', torch.zeros(367, self.hidden_dim))
    self._subtype_targets = None
    self.subtype_dn_stats = None
Decoder.__init__ = initialize


def load(solver, path):
    _load(solver, path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    source = state['ema']['module'] if 'ema' in state else state['model']
    if 'decoder.enc_score_head.fine_dn_bank' not in source:
        bank = source['decoder.denoising_class_embed.weight']
        assert bank.shape == solver.model.decoder.enc_score_head.fine_dn_bank.shape
        solver.model.decoder.enc_score_head.fine_dn_bank.copy_(bank)
    assert solver.model.decoder.enc_score_head.fine_dn_bank.abs().sum() > 0
    solver.semantic_initialization['subtype_dn'] = {'strength':STRENGTH, 'minimum_iou':.5,
        'minimum_source_probability':.3, 'bank':'frozen public source DN embeddings',
        'coarse_labels':'unchanged; no source aircraft to target UAV mapping'}
baseline.BaselineSolver.load_tuning_state = load


def encoder_input(self, feats):
    memory, shapes = _encoder(self, feats)
    targets = self._subtype_targets
    if self.training and targets is not None:
        counts = [len(t['labels']) for t in targets]
        maximum = max(counts)
        if maximum:
            # A no-grad AMP call can cache detached parameter casts and poison
            # the later differentiable call in the enclosing autocast region.
            # Keep this detached evidence path in FP32 with autocast disabled.
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                anchors, valid = self._generate_anchors(shapes, device=memory.device)
                output = self.enc_output(memory.detach().float() * valid.float())
                proposals = (self.enc_bbox_head(output) + anchors).sigmoid().float()
                proposals = box_convert(proposals, 'cxcywh', 'xyxy').clamp(0, 1)
                labels = torch.full((len(targets), maximum), self.num_classes, device=memory.device, dtype=torch.long)
                delta = memory.new_zeros(len(targets), maximum, self.hidden_dim)
                support = torch.zeros_like(labels, dtype=torch.bool)
                bank = self.enc_score_head.fine_dn_bank
                compatible_count = 0
                source_counts = {}
                for b,t in enumerate(targets):
                    count = counts[b]
                    if not count: continue
                    labels[b,:count] = t['labels']
                    truth = box_convert(t['boxes'].float(), 'cxcywh', 'xyxy')
                    overlap = box_iou(proposals[b], truth)
                    overlap.masked_fill_(~valid[0,:,0,None], -1)
                    quality, indices = overlap.max(0)
                    scores = self.enc_score_head.source(output[b,indices]).float().sigmoid()
                    probability, subtype = scores.max(-1)
                    for category, rows in pooling.GROUPS.items():
                        if len(rows) < 2: continue
                        matched = (t['labels']==category) & (quality>=.5) & (probability>=.3)
                        row_tensor = subtype.new_tensor(rows)
                        matched &= (subtype[:,None]==row_tensor).any(-1)
                        support[b,:count] |= matched
                        if matched.any():
                            selected = matched.nonzero().flatten()
                            delta[b,selected] = (bank[subtype[selected]] - bank[rows].mean(0)).to(delta.dtype)
                            compatible_count += len(selected)
                            for row in subtype[selected].tolist():
                                source_counts[str(row)] = source_counts.get(str(row), 0) + 1
                self.denoising_class_embed._subtype_context = (labels, delta, support)
                self.subtype_dn_stats = {'targets':sum(counts), 'supported':compatible_count, 'source_rows':source_counts}
    return memory, shapes
Decoder._get_encoder_input = encoder_input


def conditioned_dn(targets, num_classes, num_queries, class_embed, *args, **kwargs):
    context = getattr(class_embed, '_subtype_context', None)
    embed = class_embed if context is None else ConditionalEmbedding(class_embed, *context)
    return _dn(targets, num_classes, num_queries, embed, *args, **kwargs)
decoder_module.get_contrastive_denoising_training_group = conditioned_dn


def forward(self, feats, targets=None):
    self._subtype_targets = targets if self.training else None
    self.subtype_dn_stats = None
    try:
        return _forward(self, feats, targets)
    finally:
        self._subtype_targets = None
        if hasattr(self.denoising_class_embed, '_subtype_context'):
            del self.denoising_class_embed._subtype_context
Decoder.forward = forward
