"""Train-only ideal P3 foreground/background low-pass intervention; not deployable AP."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'D-FINE'), str(ROOT / 'scripts')]
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert, box_iou
from src.core import YAMLConfig


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / 'experiments/mechanism_trials' / f'gpu{args.gpu}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    free, total = torch.cuda.mem_get_info()
    if free < 9216 * 1024**2:
        raise RuntimeError('Insufficient free GPU memory for probe')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(8.5 * 1024**3 / total)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = YAMLConfig(str(ROOT / 'configs/ft_aug800_shared3.yml'))
    model = cfg.model
    checkpoint = ROOT / 'runs/ft_aug800/weights_epoch_020.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.cuda().eval().requires_grad_(False)
    ann = json.loads((ROOT / 'data/annotations/train1600.json').read_text())
    by_image = defaultdict(list)
    for a in ann['annotations']:
        by_image[a['image_id']].append(a)
    images = [ann['images'][i * len(ann['images']) // args.limit] for i in range(args.limit)]
    conditions = ['identity', 'bg3', 'fg3', 'all3', 'bg5', 'fg5', 'all5']
    context = {}
    records = []
    energies = []

    def hook(module, inputs, outputs):
        feature = outputs[0]
        mask = feature.new_zeros((1, 1, *feature.shape[-2:]))
        h, w = feature.shape[-2:]
        # Include a one-cell halo; compare identical masks for FG/BG interventions.
        for box in context['gt']:
            x0, y0, x1, y1 = box.tolist()
            xa, ya = max(0, int(x0*w)-1), max(0, int(y0*h)-1)
            xb, yb = min(w, int(x1*w)+2), min(h, int(y1*h)+2)
            mask[:, :, ya:yb, xa:xb] = 1
        if context['condition'] == 'identity':
            low = F.avg_pool2d(F.pad(feature, (1,1,1,1), mode='replicate'), 3, stride=1)
            high = (feature-low).square().mean(1, keepdim=True)
            energies.append({'image_id': context['image_id'], 'fg_fraction': float(mask.mean()),
                'fg_high_energy': float((high*mask).sum()/mask.sum().clamp_min(1)),
                'bg_high_energy': float((high*(1-mask)).sum()/(1-mask).sum().clamp_min(1))})
            return outputs
        k = int(context['condition'][-1])
        pad = k//2
        low = F.avg_pool2d(F.pad(feature, (pad,pad,pad,pad), mode='replicate'), k, stride=1)
        region = mask if context['condition'].startswith('fg') else 1-mask
        if context['condition'].startswith('all'):
            region = torch.ones_like(mask)
        modified = feature + .5*region*(low-feature)
        return type(outputs)([modified, *outputs[1:]])

    handle = model.backbone.register_forward_hook(hook)
    started = time.monotonic()

    def summary():
        baseline = {r['annotation_id']: r for r in records if r['condition']=='identity'}
        result = {}
        for condition in conditions:
            rows = [r for r in records if r['condition']==condition]
            groups = {}
            for group in ['small', 'regular', 'all']:
                selected = [r for r in rows if group=='all' or (r['short_side_800']<16)==(group=='small')]
                if not selected:
                    continue
                groups[group] = {'n': len(selected)}
                for metric in ['correct_score_iou50','correct_score_iou75','best_geometry_iou','argmax_class_iou']:
                    values = [r[metric] for r in selected]
                    delta = [r[metric]-baseline[r['annotation_id']][metric] for r in selected]
                    groups[group][metric] = {'mean': sum(values)/len(values), 'paired_delta': sum(delta)/len(delta),
                        'positive': sum(d>1e-4 for d in delta), 'negative': sum(d < -1e-4 for d in delta)}
            result[condition] = groups
        return result

    with torch.inference_mode():
        for index, info in enumerate(images):
            targets = by_image[info['id']]
            if not targets:
                continue
            gt = torch.tensor([a['bbox'] for a in targets], device='cuda', dtype=torch.float32)
            gt = box_convert(gt, 'xywh', 'xyxy') / gt.new_tensor([info['width'], info['height']]*2)
            context.update(gt=gt, image_id=info['id'])
            im = Image.open(ROOT / 'data/train' / info['file_name']).convert('RGB')
            tensor = TF.to_tensor(TF.resize(im, [800,800])).unsqueeze(0).cuda()
            for condition in conditions:
                context['condition'] = condition
                prediction = model(tensor)
                boxes = box_convert(prediction['pred_boxes'][0].float(), 'cxcywh','xyxy').clamp(0,1)
                probabilities = prediction['pred_logits'][0].float().sigmoid()
                matrix = box_iou(boxes, gt)
                for j, target in enumerate(targets):
                    label = target['category_id']
                    same = probabilities.argmax(-1)==label
                    row = {'image_id':info['id'], 'annotation_id':target['id'], 'category_id':label,
                        'condition':condition,'short_side_800':float((gt[j,2:]-gt[j,:2]).min()*800),
                        'best_geometry_iou':float(matrix[:,j].max()),
                        'argmax_class_iou':float(matrix[same,j].max()) if same.any() else 0.0}
                    for threshold, name in [(.5,'correct_score_iou50'),(.75,'correct_score_iou75')]:
                        eligible = matrix[:,j]>=threshold
                        row[name] = float(probabilities[eligible,label].max()) if eligible.any() else 0.0
                    records.append(row)
            report = {'state':'running','completed_images':index+1,'total_images':len(images),
                'seconds':time.monotonic()-started,'max_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
                'groups':summary(),'notes':['Official train1600 only; frozen model, no augmentation/gradients.',
                    'GT-derived mask is an ideal intervention, never deployed; these are not AP or independent validation results.',
                    'P3-only 50% low-pass residual; other backbone scales unchanged; masks include one P3-cell context halo.',
                    'Per-GT maxima are optimistic and may share queries; no one-to-one recall claim.']}
            temporary = out/'progress.json.tmp'
            temporary.write_text(json.dumps(report,indent=2))
            temporary.replace(out/'progress.json')
            if (index+1)%8==0:
                print(json.dumps({'images':index+1,'seconds':report['seconds'],'memory_mib':report['max_memory_mib']}),flush=True)
    handle.remove()
    report['state']='complete'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'records.json').write_text(json.dumps(records))
    (out/'energy.json').write_text(json.dumps(energies))
    (out/'COMPLETE').write_text('diagnosis complete; no deployment claim\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--limit',type=int,default=128)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
