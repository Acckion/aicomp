"""Paired image bootstrap of cached COCO predictions, without fitting predictions."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    assert args.samples >= 20
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.annotations))
    ids = sorted(gt.imgs)
    evaluators = {}
    hashes = {}
    for arm in ('real', 'sham'):
        path = args.predictions_root / arm / 'raw_predictions.json'
        hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        pred = json.loads(path.read_text())
        assert sorted(map(int, pred)) == ids, 'Must cover the complete same heldout fold'
        rows = []
        for iid, p in pred.items():
            for b, s, c in zip(p['boxes'], p['scores'], p['labels']):
                rows.append(dict(image_id=int(iid), category_id=int(c), bbox=[b[0], b[1], b[2]-b[0], b[3]-b[1]], score=s))
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(rows), 'bbox')
            ev.params.imgIds = ids
            ev.evaluate()
            ev.accumulate()
        evaluators[arm] = ev
    n = len(ids)
    groups = len(evaluators['real'].params.catIds) * len(evaluators['real'].params.areaRng)
    assert all(len(ev.evalImgs) == groups*n for ev in evaluators.values())
    arrays = {arm: np.asarray(ev.evalImgs, dtype=object).reshape(groups, n) for arm, ev in evaluators.items()}
    def ap(ev, indices, arm):
        replica = copy.copy(ev)
        replica.params = copy.deepcopy(ev.params)
        replica._paramsEval = copy.deepcopy(ev._paramsEval)
        # Virtual image slots let repeated images contribute independently.
        # Matching remains the original within-image, one-to-one COCO matching.
        replica.params.imgIds = list(range(n))
        replica._paramsEval.imgIds = list(range(n))
        replica.evalImgs = arrays[arm][:, indices].reshape(-1).tolist()
        with contextlib.redirect_stdout(io.StringIO()):
            replica.accumulate()
        precision = replica.eval['precision'][:, :, :, 0, -1]
        class_ap = []
        for category in range(precision.shape[2]):
            values = precision[:, :, category]
            values = values[values >= 0]
            class_ap.append(float(values.mean()*100) if values.size else None)
        return class_ap
    identity = np.arange(n)
    original = {}
    for arm, ev in evaluators.items():
        observed = ap(ev, identity, arm)
        values = ev.eval['precision'][:, :, :, 0, -1]
        expected = [float(values[:, :, k][values[:, :, k] >= 0].mean()*100) for k in range(values.shape[2])]
        assert np.allclose(observed, expected, atol=1e-10), 'Identity resampling must reproduce original AP'
        original[arm] = observed
    rng = np.random.default_rng(args.seed)
    samples = []
    for _ in range(args.samples):
        indices = rng.integers(0, n, n)
        results = {}
        for arm, ev in evaluators.items():
            results[arm] = ap(ev, indices, arm)
        present = [k for k, value in enumerate(results['real']) if value is not None]
        assert present == [k for k, value in enumerate(results['sham']) if value is not None]
        delta = [results['real'][k]-results['sham'][k] for k in present]
        samples.append({'present_classes': len(present), 'macro_delta': float(np.mean(delta))})
    complete = [s['macro_delta'] for s in samples if s['present_classes'] == len(original['real'])]
    assert complete, 'No resample has support for all official categories'
    report = {
        'scope': 'Paired image bootstrap of complete heldout cached predictions; no model fitting or phase2 claim.',
        'limitations': 'Images are the resampling units. Correlation across images of the same physical scene may make intervals optimistic. All-category interval is conditional on resamples containing every category; absent-category draws are reported separately. Finite bootstrap is a diagnostic, not proof of generalization.',
        'images': n, 'samples': args.samples, 'seed': args.seed,
        'prediction_sha256': hashes,
        'annotations_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        'identity_resampling_exact': True,
        'observed_macro_delta': float(np.mean(original['real'])-np.mean(original['sham'])),
        'complete_category_samples': len(complete),
        'conditional_95pct_interval': np.quantile(complete, [.025, .975]).tolist(),
        'conditional_positive_fraction': float(np.mean(np.asarray(complete)>0)),
        'replicates': samples,
    }
    path = args.predictions_root / 'paired_bootstrap.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='replicates'}, indent=2))


if __name__ == '__main__':
    main()
