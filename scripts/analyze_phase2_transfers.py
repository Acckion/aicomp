"""Compare scored submission predictions, without test labels or training."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/phase2_transfer_audit'
OUT.mkdir(parents=True,exist_ok=True)


def iou(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(br-tl,0).prod(-1)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(-1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)


def main():
    images=json.loads((ROOT/'data/phase2/images.json').read_text())['images']
    train=json.loads((ROOT/'data/annotations/train2000.json').read_text())
    names={c['id']:c['name'] for c in train['categories']}
    packages={
        'rgb2000_flip_soft':(ROOT/'experiments/light_tuning/phase2_flip_gaussian07/submission.zip',49.986),
        'ir2000_native':(ROOT/'checkpoints/gpu6_storage/submissions/full2000_ir_ep006_800/submission.zip',49.752),
        'ir1600_gateoff_native':(ROOT/'experiments/tonight_20261002_validated_ir/submission.zip',49.136),
    }
    all_pred={};summary={}
    for model,(path,score) in packages.items():
        records={};classes=defaultdict(lambda: {'all':0,'ge_025':0,'ge_050':0,'small_short_side_800_lt16':0,'scores':[]})
        with zipfile.ZipFile(path) as archive:
            assert len(archive.namelist())==1000 and archive.testzip() is None
            for im in images:
                rows=np.asarray([list(map(float,l.split())) for l in archive.read(Path(im['file_name']).stem+'.txt').decode().splitlines()],dtype=float).reshape(-1,6)
                assert len(rows)<=100 and np.isfinite(rows).all()
                records[im['id']]=rows
                for c,x,y,w,h,s in rows:
                    p=classes[int(c)];p['all']+=1;p['ge_025']+=int(s>=.25);p['ge_050']+=int(s>=.5)
                    p['small_short_side_800_lt16']+=int(min(w,h)*800<16);p['scores'].append(float(s))
        all_pred[model]=records
        summary[model]={'official_score':score,'classes':{names[c]:{**{k:v for k,v in p.items() if k!='scores'},'score_quantiles':np.quantile(p['scores'],[.1,.5,.9]).tolist()} for c,p in classes.items()}}
    comparison=defaultdict(lambda:{'old_highconf':0,'new_highconf':0,'matched_iou50':0,'matched_iou75':0,'ious':[],'score_delta_new_old':[]})
    examples=[]
    for im in images:
        old=all_pred['ir2000_native'][im['id']];new=all_pred['ir1600_gateoff_native'][im['id']]
        def xyxy(r):return np.stack([r[:,1]-r[:,3]/2,r[:,2]-r[:,4]/2,r[:,1]+r[:,3]/2,r[:,2]+r[:,4]/2],-1)
        for c in range(12):
            a=old[(old[:,0]==c)&(old[:,5]>=.25)];b=new[(new[:,0]==c)&(new[:,5]>=.25)]
            p=comparison[names[c]];p['old_highconf']+=len(a);p['new_highconf']+=len(b)
            matrix=iou(xyxy(a),xyxy(b))
            pairs=np.argwhere(matrix>=.5);used_a=set();used_b=set()
            for ai,bi in sorted(pairs,key=lambda ij:-matrix[tuple(ij)]):
                if int(ai) in used_a or int(bi) in used_b:continue
                used_a.add(int(ai));used_b.add(int(bi));overlap=float(matrix[ai,bi])
                p['matched_iou50']+=1;p['matched_iou75']+=int(overlap>=.75);p['ious'].append(overlap)
                p['score_delta_new_old'].append(float(b[bi,5]-a[ai,5]))
            for bi,r in enumerate(b):
                best=float(matrix[:,bi].max()) if len(a) else 0
                if bi not in used_b and r[5]>=.65:
                    examples.append({'image':im,'class':names[c],'row':r.tolist(),'best_old_iou':best})
    compact={}
    for c,p in comparison.items():
        compact[c]={**{k:v for k,v in p.items() if k not in ('ious','score_delta_new_old')},'matched_iou_mean':float(np.mean(p['ious'])) if p['ious'] else None,'matched_score_delta':float(np.mean(p['score_delta_new_old'])) if p['ious'] else None}
    report={'submissions':summary,'ir_pair_prediction_agreement':compact,
            'limitations':'No test GT; prediction disagreement or confidence changes cannot identify false positives, misses, class AP or the cause of official score changes. IR1600 and IR2000 differ in sample count and training settings. Visual review is descriptive only, never used to alter test predictions or train.',
            'new_highconfidence_unmatched_count':len(examples)}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    chosen=[];seen=Counter()
    for e in sorted(examples,key=lambda e:-e['row'][5]):
        if seen[e['class']]>=2:continue
        chosen.append(e);seen[e['class']]+=1
        if len(chosen)==12:break
    canvas=Image.new('RGB',(1200,4*245),'white');draw=ImageDraw.Draw(canvas)
    for k,e in enumerate(chosen):
        im=e['image'];r=e['row'];x=(k%3)*400;y=(k//3)*245
        with Image.open(ROOT/'data/phase2'/im['file_name']) as image:
            image=image.convert('RGB');w,h=image.size
            cx,cy,bw,bh=r[1:5];pad=.15
            crop=image.crop((max(0,(cx-bw*(.5+pad))*w),max(0,(cy-bh*(.5+pad))*h),min(w,(cx+bw*(.5+pad))*w),min(h,(cy+bh*(.5+pad))*h)))
            crop.thumbnail((398,210));canvas.paste(crop,(x,y+32))
        draw.text((x+3,y+4),f"{e['class']} score={r[5]:.2f} oldIoU={e['best_old_iou']:.2f}",fill='black')
        draw.text((x+3,y+18),Path(im['file_name']).name,fill='black')
    canvas.save(OUT/'new_unmatched_review.jpg')
    print(json.dumps({'comparison':compact,'new_highconfidence_unmatched_count':len(examples)}))


if __name__=='__main__':main()
