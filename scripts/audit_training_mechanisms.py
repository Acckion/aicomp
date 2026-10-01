"""Reproduce training-only similarity and BatchNorm-state diagnostics."""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/mechanism_audit'


def prepare_bn_candidates():
    import torch
    checkpoint=torch.load(ROOT/'runs/contextfull800/last.pth',map_location='cpu',weights_only=False)
    parent=torch.load(ROOT/'runs/ft_aug800/weights_epoch_020.pth',map_location='cpu',weights_only=False)['model']
    ema=checkpoint['ema']['module'];transplanted={k:v.clone() for k,v in ema.items()}
    keys=[k for k in ema if k.endswith(('running_mean','running_var','num_batches_tracked')) and not k.startswith('backbone.')]
    for k in keys:transplanted[k]=parent[k].clone()
    for name,weights in [('context_raw',checkpoint['model']),('context_parent_bn',transplanted)]:
        torch.save({'model':weights},OUT/(name+'.pth'))
    updates=checkpoint['ema']['updates']
    report={'swapped_non_backbone_bn_buffers':len(keys),'ema_updates':updates,
            'ema_actual_final_decay':.999*(1-math.exp(-updates/1000)),
            'bn_buffer_keys':keys,'note':'Validation-only ablation; learned EMA parameters unchanged in BN transplant. Not a submission model.'}
    (OUT/'bn_audit.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='bn_buffer_keys'})


def prepare_deim_bn_candidate():
    import torch
    from safetensors.torch import load_file
    weights=torch.load(ROOT/'runs/deimv2_x/last.pth',map_location='cpu',weights_only=False)['ema']['module']
    parent=load_file(str(ROOT/'checkpoints/deimv2/X_model.safetensors'))
    keys=[k for k in weights if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    for k in keys:weights[k]=parent[k].clone()
    torch.save({'model':weights},OUT/'deimx_parent_bn.pth')
    print('DEIM X public COCO BN buffers restored:',len(keys))


def similarity():
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn
    all_images=json.loads((ROOT/'data/annotations/train2000.json').read_text())['images']
    train=json.loads((ROOT/'data/annotations/train1600.json').read_text())['images']
    val=json.loads((ROOT/'data/annotations/val400.json').read_text())['images']
    def encode(im):
        with Image.open(ROOT/'data/train'/im['file_name']) as image:
            image.draft('RGB',(128,128));tiny=image.convert('RGB').resize((32,32))
            rgb=np.asarray(tiny,dtype=np.float32)/255
            low=dctn(np.asarray(tiny.convert('L'),dtype=float),norm='ortho')[:8,:8].reshape(-1)[1:]
            bits=np.packbits(low>np.median(low))
        return im['id'],bits,rgb
    with concurrent.futures.ThreadPoolExecutor(8) as pool:encoded=list(pool.map(encode,all_images))
    features={i:(bits,rgb) for i,bits,rgb in encoded}
    train_ids=[im['id'] for im in train];train_hash=np.array([features[i][0] for i in train_ids])
    lut=np.array([i.bit_count() for i in range(256)])
    rows=[]
    for im in val:
        bits,rgb=features[im['id']];distance=lut[np.bitwise_xor(train_hash,bits)].sum(1)
        idx=int(distance.argmin());iid=train_ids[idx]
        rows.append({'val_image_id':im['id'],'train_image_id':iid,'phash_hamming63':int(distance[idx]),
                     'rgb_rmse32':float(np.sqrt(np.mean((rgb-features[iid][1])**2)))})
    report={'images':len(rows),'nearest_hamming_median':float(np.median([r['phash_hamming63'] for r in rows])),
            'counts_at_hamming':{str(k):sum(r['phash_hamming63']<=k for r in rows) for k in [2,4,8,12]},
            'strict_candidates_hamming4_rmse005':sum(r['phash_hamming63']<=4 and r['rgb_rmse32']<=.05 for r in rows),
            'pairs':sorted(rows,key=lambda r:(r['phash_hamming63'],r['rgb_rmse32'])),
            'note':'Train-only approximate similarity; visually review candidates. No test data or replacement split.'}
    (OUT/'split_similarity.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='pairs'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--similarity',action='store_true');parser.add_argument('--bn-candidates',action='store_true');parser.add_argument('--deimx-bn-candidate',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.similarity:similarity()
    if args.bn_candidates:prepare_bn_candidates()
    if args.deimx_bn_candidate:prepare_deim_bn_candidate()
