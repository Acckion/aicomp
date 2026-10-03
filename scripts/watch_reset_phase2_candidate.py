"""Prepare a fixed-epoch reset-head candidate after independent evidence gates.

Reuse the existing audited package writer; this is not an automatic submission
or permission to change the pool candidate's acceptance rule.
"""
import json
import os
from pathlib import Path

import watch_taxonomy_phase2_candidate as package

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/phase2_obj365_reset_e40'


def comparison():
    reset = package.records(ROOT/'runs/scene_obj365_reset800/metrics.jsonl')
    control = package.records(ROOT/'runs/scene_rgb_fastcontrol800/metrics.jsonl')
    epochs = [e for e in sorted(reset.keys() & control.keys()) if e >= 4][-3:]
    if len(epochs) < 3:
        return None
    deltas, nonsparse, ap75 = [], [], []
    for epoch in epochs:
        a, b = reset[epoch]['validation'], control[epoch]['validation']
        deltas.append(100*(a['coco_eval_bbox'][0]-b['coco_eval_bbox'][0]))
        ap75.append(100*(a['coco_eval_bbox'][2]-b['coco_eval_bbox'][2]))
        nonsparse.append(sum(100*(value-b['per_class_ap'][category])
            for category,value in a['per_class_ap'].items() if category!='tricycle')/11)
    return {'comparison':'scene_obj365_reset800 vs scene_rgb_fastcontrol800',
            'common_epochs':epochs,'map_deltas':deltas,
            'mean_map_delta':sum(deltas)/3,
            'mean_ap75_delta':sum(ap75)/3,
            'mean_delta_excluding_sparse_tricycle':sum(nonsparse)/3,
            'supported':sum(deltas)/3 >= .5 and sum(nonsparse)/3 > 0,
            'limitations':'Matched seed and fold, one training realization. '
                'Public pretraining differs; this is a candidate comparison, not isolated classification-head causality or online gain.'}


def configure():
    report_path = ROOT/'experiments/taxonomy_matched_e30/report.json'
    report = json.loads(report_path.read_text())
    identity = json.loads((report_path.parent/'reset/cache_identity.json').read_text())
    result = report['results']['reset']
    assert report['epoch']==30 and report['images']==390
    assert identity['checkpoint_sha256']==report['checkpoint_provenance']['reset']['sha256']
    assert abs(result['txt_roundtrip_delta']) < 1e-8
    assert len(json.loads((ROOT/'data/annotations/scene_val.json').read_text())['images'])==390
    OUT.mkdir(parents=True,exist_ok=True)
    audit = OUT/'validation_route_audit.json'
    audit.write_text(json.dumps({'route_consistent':True,
        'source_report':str(report_path),'source_checkpoint':report['checkpoint_provenance']['reset'],
        'epoch':30,'images':390,'standalone_native_map':result['map'],
        'txt_map':result['txt_roundtrip_map'],'txt_roundtrip_delta':result['txt_roundtrip_delta'],
        'limitations':'Independent reset shadow epoch30 with native800 top100 strict EMA; full2000 has no independent validation.'},indent=2)+'\n')
    package.OUT = OUT
    package.RUN = ROOT/'runs/full2000_obj365_reset800_b2'
    package.CONFIG = ROOT/'configs/full2000_obj365_reset800_b2.yml'
    package.AUDIT = audit
    package.evidence = comparison


if __name__=='__main__':
    configure()
    (OUT/'controller.pid').write_text(str(os.getpid())+'\n')
    package.main()
