"""TIDE idealized error interventions on independent validation predictions only."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'D-FINE'))
from evaluate_variants import select,coco_rows
from tidecv import TIDE,Data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',default=str(ROOT/'experiments/next_stage/infer800_tile0/raw_predictions.json'))
    parser.add_argument('--output',default=str(ROOT/'experiments/mechanism_audit/tide'))
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ann=ROOT/'data/annotations/val400.json';raw=json.loads(Path(args.predictions).read_text())
    data=json.loads(ann.read_text());assert set(raw)=={str(i['id']) for i in data['images']}
    results=coco_rows({i:select(p) for i,p in raw.items()})
    (out/'coco_predictions.json').write_text(json.dumps(results))
    # Box-only AICOMP COCO annotations intentionally have no segmentation.
    gt=Data('val400');pred=Data('baseline800',max_dets=100)
    for c in data['categories']:gt.add_class(c['id'],c['name']);pred.add_class(c['id'],c['name'])
    for im in data['images']:gt.add_image(im['id'],im['file_name']);pred.add_image(im['id'],im['file_name'])
    for a in data['annotations']:gt.add_ground_truth(a['image_id'],a['category_id'],box=a['bbox'])
    for p in results:pred.add_detection(p['image_id'],p['category_id'],p['score'],box=p['bbox'])
    rows=[]
    for k in range(10):
        threshold=.5+.05*k;tide=TIDE();run=tide.evaluate(gt,pred,pos_threshold=threshold,mode=TIDE.BOX,name='baseline800')
        errors=tide.get_all_errors()
        row={'iou':threshold,'ap':run.ap,'main':errors['main']['baseline800'],'special':errors['special']['baseline800']}
        rows.append(row);print(json.dumps(row),flush=True)
    report={'rows':rows,'mean_ap':statistics.mean(r['ap'] for r in rows),
            'mean_idealized_ap_gain':{name:statistics.mean(r['main'][name] for r in rows) for name in rows[0]['main']},
            'notes':['TIDE separate idealized error fixes; gains are not additive or attainable training gains.',
                     'Independent val400 only. No predictions corrected for submission or test.',
                     'Source predictions clipped and top100 per image, unchanged classes/coordinates.']}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['mean_idealized_ap_gain']),flush=True)


if __name__=='__main__':main()
