"""Cluster-bootstrap train-only diagnostic deltas; never a validation AP claim."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(directory):
    directory = Path(directory)
    rows = json.loads((directory/'records.json').read_text())
    baseline = {r['annotation_id']:r for r in rows if r['condition']=='identity'}
    metrics = ['correct_score_iou50','correct_score_iou75','best_geometry_iou','argmax_class_iou']
    report = {}
    for condition in ['bg3','bg5','fg3','fg5','all3','all5']:
        report[condition] = {}
        for group in ['small','regular','all']:
            selected = [r for r in rows if r['condition']==condition and (group=='all' or (r['short_side_800']<16)==(group=='small'))]
            ids = sorted({r['image_id'] for r in selected})
            lookup = {i:j for j,i in enumerate(ids)}
            count = np.zeros(len(ids))
            sums = np.zeros((len(ids),len(metrics)))
            for r in selected:
                index = lookup[r['image_id']]
                count[index] += 1
                sums[index] += [r[m]-baseline[r['annotation_id']][m] for m in metrics]
            rng = np.random.default_rng(20261002)
            resampled = rng.integers(0,len(ids),size=(1000,len(ids)))
            values = sums[resampled].sum(1)/count[resampled].sum(1)[:,None]
            report[condition][group] = {'n_gt':len(selected),'n_images':len(ids), 'deltas':{
                m:{'mean':float(sums[:,j].sum()/count.sum()),
                   'image_cluster_bootstrap_ci95':np.quantile(values[:,j],[.025,.975]).tolist()} for j,m in enumerate(metrics)}}
    result = {'paired_diagnostics':report,'seed':20261002,'bootstrap_draws':1000,
        'decision':'No deployable training is automatically justified by GT-masked train-only interventions. Inspect the confidence intervals and both IoU thresholds.',
        'limitations':['Not independent validation, not AP; frozen-network distribution interventions.',
            'Tests P3-only 50%-strength box/halo lowpass, not the complete SET mechanism.']}
    (directory/'bootstrap.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    main(parser.parse_args().directory)
