"""Original-image tile evidence enters one detector before query selection.

This is feature fusion, not tiled box merging. A dense-grid control distinguishes
new pixel evidence from interpolating the mature feature grid. Crops never use GT.
"""
from copy import deepcopy
import fcntl
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import native_roi  # Reuse original-image packing, not its ROI refinement.
import train_baseline as baseline
from src.core import register
from src.zoo.dfine.dfine import DFINE


class DetailBridge(nn.Module):
    def __init__(self, projection, hidden_dim):
        super().__init__()
        self.projection = deepcopy(projection)
        self.output = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, x):
        # Mature BN running statistics stay fixed for single-tile inference.
        self.projection.eval()
        return self.output(self.projection(x))


@register()
class NativeGridDFINE(DFINE):
    def __init__(self, backbone, encoder, decoder, size=800,
                 detail_enabled=True, grid_factor=1, tile_size=800):
        super().__init__(backbone, encoder, decoder)
        assert grid_factor in (1, 2)
        self.size = size
        self.tile_size = tile_size
        self.detail_enabled = detail_enabled
        self.grid_factor = grid_factor
        self.detail_mode = 'real'
        self.forced_size = None
        self.detail_bridge = DetailBridge(encoder.input_proj[0], encoder.hidden_dim)
        # Decoder anchors must follow actual feature sizes, including the
        # denser first level; the encoder itself keeps its original hierarchy.
        self.decoder.eval_spatial_size = None
        self.encoder.eval_spatial_size = None

    def native_features(self, packed, canvas_shape):
        maps = []
        training_flags = [(m, m.training) for m in self.backbone.modules()]
        self.backbone.eval()  # No regional BN update or dropout.
        for image in packed:
            h, w = map(int, image[6, 0, :2].detach().cpu().tolist())
            assert h > 0 and w > 0
            native = image[3:6, :h, :w][None]
            if self.detail_mode == 'global':
                # Content control: discard original detail before making crops.
                native = F.interpolate(image[None, :3, :self.size, :self.size],
                                       size=(h, w), mode='bilinear', align_corners=False)
            crop_h, crop_w = round(h * .6), round(w * .6)
            assembled = None
            coverage = native.new_zeros((1, 1, *canvas_shape))
            for y in (0, h-crop_h):
                for x in (0, w-crop_w):
                    tile = F.interpolate(native[:, :, y:y+crop_h, x:x+crop_w],
                                         size=(self.tile_size, self.tile_size),
                                         mode='bilinear', align_corners=False)
                    with torch.no_grad():
                        feature = self.backbone(tile)[0].detach()
                    # Reconstruct every crop in global normalized coordinates.
                    ch, cw = canvas_shape
                    ya, yb = round(y/h*ch), round((y+crop_h)/h*ch)
                    xa, xb = round(x/w*cw), round((x+crop_w)/w*cw)
                    feature = F.interpolate(feature, size=(yb-ya, xb-xa),
                                            mode='bilinear', align_corners=False)
                    if assembled is None:
                        assembled = feature.new_zeros((1, feature.shape[1], ch, cw))
                    assembled[:, :, ya:yb, xa:xb] += feature
                    coverage[:, :, ya:yb, xa:xb] += 1
            assert (coverage > 0).all()
            assembled = assembled / coverage
            if self.detail_mode == 'zero':
                assembled = torch.zeros_like(assembled)
            elif self.detail_mode == 'shuffle':
                flat = assembled.flatten(2)
                assembled = flat[:, :, torch.randperm(flat.shape[-1], device=flat.device)].reshape_as(assembled)
            maps.append(assembled)
        for module, flag in training_flags:
            module.training = flag
        return torch.cat(maps)

    def forward(self, packed, targets=None):
        assert packed.shape[1] == 7
        rgb = packed[:, :3, :self.size, :self.size]
        active_size = self.forced_size or self.size
        if self.training and self.forced_size is None:
            sizes = (640, 704, 768, 800, 832, 896, 960, 992)
            active_size = sizes[int(torch.randint(len(sizes), ()).item())]
        if active_size != self.size:
            rgb = F.interpolate(rgb, size=(active_size, active_size), mode='bilinear', align_corners=False)
        features = self.encoder(self.backbone(rgb))
        features = list(features)
        canvas = tuple(s * self.grid_factor for s in features[0].shape[-2:])
        if self.grid_factor != 1:
            features[0] = F.interpolate(features[0], size=canvas, mode='bilinear', align_corners=False)
        if self.detail_enabled:
            feature = self.native_features(packed, canvas)
            features[0] = features[0] + self.detail_bridge(feature).to(features[0].dtype)
        return self.decoder(features, targets)


def parent_weights(path):
    source = Path(path)
    signature = {'path': str(source.resolve()), 'bytes': source.stat().st_size,
                 'mtime_ns': source.stat().st_mtime_ns}
    cache = Path('/dev/shm/aicomp_native_grid_parent.pth')
    with Path('/dev/shm/aicomp_native_grid_parent.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            cached = torch.load(cache, map_location='cpu', weights_only=False)
            if cached.get('signature') == signature:
                return cached['model']
        state = torch.load(source, map_location='cpu', weights_only=False)
        weights = state['ema']['module'] if 'ema' in state else state['model']
        tmp = cache.with_suffix('.tmp')
        torch.save({'signature': signature, 'model': weights}, tmp)
        tmp.replace(cache)
        return weights


def initialize(model, path):
    weights = dict(parent_weights(path))
    current = model.state_dict()
    for key in ('decoder.anchors', 'decoder.valid_mask'):
        if key in current:
            weights[key] = current[key]
    result = model.load_state_dict(weights, strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('detail_bridge.') for k in result.missing_keys), result
    model.detail_bridge.projection.load_state_dict(model.encoder.input_proj[0].state_dict(), strict=True)
    assert torch.count_nonzero(model.detail_bridge.output.weight) == 0


_original_load = baseline.BaselineSolver.load_tuning_state
def load(solver, path):
    if not isinstance(solver.model, NativeGridDFINE):
        return _original_load(solver, path)
    initialize(solver.model, path)
    print('NATIVE_GRID_PARENT mature RGB loaded; projection copied; residual zero', flush=True)
baseline.BaselineSolver.load_tuning_state = load
