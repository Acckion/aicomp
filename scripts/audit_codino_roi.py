"""Compare individual trained Co-DINO heads; never fuse their predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import train_codino_aic as transfer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-cap-gib', type=float, default=3.5)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    transfer.OUT = args.output
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = transfer.configure(transfer.Config.fromfile(str(transfer.ROOT/'configs/codino_aic.py')))
    model = transfer.build(cfg, str(args.checkpoint), partial=False).eval()
    load = json.loads((args.output/'pretrained_load.json').read_text())
    assert not load['unmatched'] and not load['class_transferred']
    assert load['copied_keys'] == load['model_keys'], 'Every trained checkpoint tensor must match'
    train = transfer.build_dataset(cfg.data.train)
    val = transfer.build_dataset(cfg.data.val)
    assert len(train) == 1600 and len(val) == 400
    assert set(train.img_ids).isdisjoint(val.img_ids)
    loader = transfer.build_dataloader(val, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    report = {'scope':'Same trained Co-DINO checkpoint and original heldout400; individual detection heads, no fusion or phase2 prediction.',
        'checkpoint_sha256':hashlib.file_digest(args.checkpoint.open('rb'), 'sha256').hexdigest(),
        'annotations_sha256':hashlib.sha256(Path(cfg.data.val.ann_file).read_bytes()).hexdigest(),
        'memory_cap_gib':args.memory_cap_gib, 'strict_tensor_coverage':True,
        'limitations':'Original fold is not scene-isolated. Co-DINO was trained with collaborative heads and partial backbone adaptation. This audits its existing ROI path, not an independently trained Cascade-RCNN.', 'results':{}}
    for mode in ('detr', 'two-stage'):
        start = time.time()
        model.eval_module = mode
        model.eval_index = 0
        results = []
        for index, data in enumerate(loader):
            predictions = model(return_loss=False, rescale=True, **transfer.batch_cuda(data))
            assert len(predictions) == 1
            assert len(predictions[0]) == 12
            assert sum(len(p) for p in predictions[0]) <= 100
            results.extend(predictions)
            if index % 25 == 0:
                print(json.dumps({'mode':mode, 'images':len(results), 'allocated_mib':torch.cuda.memory_allocated()/2**20}), flush=True)
            if args.smoke and len(results) == 2:
                break
        if args.smoke:
            report['results'][mode] = {'images':len(results), 'finite':all(torch.isfinite(torch.as_tensor(p)).all().item() for r in results for p in r)}
            assert report['results'][mode]['finite']
            continue
        assert len(results) == len(val)
        files, _ = val.format_results(results, str(args.output/(mode+'_predictions')))
        ev = transfer.COCOeval(val.coco, val.coco.loadRes(files['bbox']), 'bbox')
        ev.params.imgIds = val.img_ids
        ev.params.maxDets = [1,10,100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        per_class = {}
        for k, category in enumerate(ev.params.catIds):
            values = ev.eval['precision'][:,:,k,0,-1]
            values = values[values >= 0]
            per_class[val.coco.cats[category]['name']] = float(values.mean()*100) if values.size else None
        report['results'][mode] = {'images':len(results), 'map':float(ev.stats[0]*100),
            'ap75':float(ev.stats[2]*100), 'small_ap':float(ev.stats[3]*100),
            'stats':[float(s*100) for s in ev.stats], 'per_class':per_class, 'seconds':time.time()-start}
        transfer.atom(args.output/'report.json', report)
    report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
    filename = 'smoke.json' if args.smoke else 'report.json'
    transfer.atom(args.output/filename, report)
    if not args.smoke:
        transfer.atom(args.output/'COMPLETE', {'complete':True})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
