"""Numerical checks of assignment, dense gradients and native export."""
import json
import torch
import train_baseline
import aux_detection
from src.core import YAMLConfig
from mechanism_runtime import freeze_bn_statistics


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024**3 /
                                             torch.cuda.get_device_properties(0).total_memory)
    device = 'cuda'
    head = aux_detection.DenseAuxHead(32).to(device)
    features = [torch.randn(1, 32, size, size, device=device, requires_grad=True)
                for size in [16, 8, 4]]
    targets = [{'boxes': torch.tensor([[.4, .4, .25, .25], [.8, .8, .005, .005]], device=device),
                'labels': torch.tensor([0, 2], device=device)}]
    loss = head(features, targets)
    sum(loss.values()).backward()
    assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in features)
    assert head.reg.weight.grad.abs().sum() > 0 and head.cls.weight.grad.abs().sum() > 0
    assert head.last_statistics['positive_locations'] >= 4
    assert head.last_statistics['multi_positive_gt'] == 2
    statistics = head.last_statistics.copy()
    empty = [{'boxes': torch.empty(0, 4, device=device), 'labels': torch.empty(0, dtype=torch.long, device=device)}]
    assert all(torch.isfinite(x) for x in head(features, empty).values())
    assert head.last_statistics['gt_count'] == 0
    cfg = YAMLConfig('/home/fbohan/AIC/configs/aux_detection800.yml')
    model = cfg.model.to(device)
    source = torch.load('/home/fbohan/AIC/runs/ft_aug800/weights_epoch_020.pth',
                        map_location='cpu', weights_only=False)['model']
    current = model.state_dict()
    source['decoder.anchors'] = current['decoder.anchors']
    source['decoder.valid_mask'] = current['decoder.valid_mask']
    missing, unexpected = model.load_state_dict(source, strict=False)
    assert not unexpected and all(k.startswith('aux_head.') for k in missing)
    model.train(); freeze_bn_statistics(model)
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        feats = model.encoder(model.backbone(torch.rand(1, 3, 192, 192, device=device)))
        auxiliary_loss = sum(model.aux_head(feats, targets).values())
    auxiliary_loss.backward()
    norms = {}
    for prefix in ['backbone.', 'encoder.', 'aux_head.']:
        norms[prefix] = sum(float(p.grad.float().norm()) for n, p in model.named_parameters()
                            if n.startswith(prefix) and p.grad is not None)
        assert norms[prefix] > 0 and torch.isfinite(torch.tensor(norms[prefix]))
    assert not any(p.grad is not None for n, p in model.named_parameters() if n.startswith('decoder.'))
    native_cfg = YAMLConfig('/home/fbohan/AIC/configs/aux_detection_control800.yml')
    native = native_cfg.model
    exported = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('aux_head.')}
    native.load_state_dict(exported, strict=True)
    # The eval graph bypasses the auxiliary module entirely.
    called = [False]
    handle = model.aux_head.register_forward_hook(lambda *args: called.__setitem__(0, True))
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        output = model(torch.rand(1, 3, 800, 800, device=device))
    handle.remove()
    assert not called[0] and 'dense_training_losses' not in output
    assert torch.isfinite(output['pred_boxes']).all()
    print('AUX_NUMERICAL_CHECK_PASSED '+json.dumps({'synthetic_assignment': statistics,
          'aux_only_gradient_norms': norms, 'native_strict_export': True,
          'aux_head_called_during_eval': called[0],
          'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}), flush=True)


if __name__ == '__main__': main()
