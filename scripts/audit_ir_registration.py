"""Training-only cross-spectral translation diagnostic, never a calibration.

Edge phase correlation can match unrelated structures. Wrong-pair controls and
synthetic shifts expose that failure mode; estimates must not warp labels.
"""
import argparse
import io
import json
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def edges(image):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (480, 270), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    # Fixed central crop reduces the influence of IR black borders.
    return edge[22:248, 72:408]


def estimate(a, b):
    """Return translation to apply to b to match a, in 480x270 pixels."""
    h, w = a.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fa * fb.conj()
    corr = np.fft.fftshift(np.fft.ifft2(cross / np.maximum(abs(cross), 1e-8)).real)
    cy, cx = h // 2, w // 2
    radius_x, radius_y = 72, 32
    search = corr[cy-radius_y:cy+radius_y+1, cx-radius_x:cx+radius_x+1]
    py, px = np.unravel_index(search.argmax(), search.shape)
    dy, dx = int(py-radius_y), int(px-radius_x)
    sidelobe = np.ones(search.shape, bool)
    sidelobe[max(0,py-3):py+4, max(0,px-3):px+4] = False
    psr = (search[py,px] - search[sidelobe].mean()) / max(search[sidelobe].std(), 1e-8)
    # Compare identity and translated edges on exactly the same overlap.
    ya, yb = max(0, dy), max(0, -dy)
    xa, xb = max(0, dx), max(0, -dx)
    hh, ww = h-abs(dy), w-abs(dx)
    ref = a[ya:ya+hh, xa:xa+ww]
    moved = b[yb:yb+hh, xb:xb+ww]
    original = b[ya:ya+hh, xa:xa+ww]
    def similarity(x, y):
        x, y = x-x.mean(), y-y.mean()
        return float((x*y).sum() / max(np.sqrt((x*x).sum()*(y*y).sum()), 1e-8))
    return {'dx': dx, 'dy': dy, 'psr': float(psr),
            'edge_correlation_identity': similarity(ref, original),
            'edge_correlation_shifted': similarity(ref, moved),
            'boundary_peak': abs(dx)==radius_x or abs(dy)==radius_y}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=128)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/ir_registration_train')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    annotation = ROOT/'data/annotations/scene_train.json'
    data = json.loads(annotation.read_text())
    selected = random.Random(20261003).sample(sorted(data['images'],key=lambda x:x['id']),
                                            min(args.sample,len(data['images'])))
    archive_path = ROOT/'初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000.zip'
    pairs = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {(Path(n).parts[-2],Path(n).name): n for n in archive.namelist()
                   if len(Path(n).parts)>1 and not n.endswith('/')}
        for meta in selected:
            name = Path(meta['file_name']).name
            views = []
            for mode in ['visible','infrared']:
                with Image.open(io.BytesIO(archive.read(members[mode,name]))) as im:
                    views.append(im.convert('RGB').resize((480,270)))
            pairs.append((meta,views,edges(views[0]),edges(views[1])))
    if len(pairs)<2:
        raise ValueError('Need at least two images for wrong-pair control')
    # A cyclic permutation has no self-pairs; similar scenes can remain.
    rows = []
    synthetic = []
    for i,(meta,views,a,b) in enumerate(pairs):
        paired = estimate(a,b)
        wrong = estimate(a,pairs[(i+1)%len(pairs)][3])
        rows.append({'image_id':meta['id'],'filename':Path(meta['file_name']).name,
                     'paired':paired,'wrong_pair':wrong,
                     'paired_abs_displacement_800':float(np.hypot(paired['dx']*800/480,paired['dy']*800/270))})
        for dx,dy in [(7,-5),(-11,9)]:
            transformed = np.roll(a,(dy,dx),axis=(0,1))
            estimated = estimate(a,transformed)
            synthetic.append({'image_id':meta['id'],'applied_shift':[dx,dy],
                              'estimated_inverse':[estimated['dx'],estimated['dy']],
                              'error_pixels':float(np.hypot(estimated['dx']+dx,estimated['dy']+dy))})
        print(json.dumps({'images_done':i+1,'total':len(pairs)}),flush=True)
    def quantiles(values):
        return dict(zip(['p10','p50','p90'],map(float,np.quantile(values,[.1,.5,.9]))))
    report = {'scope':'scene_train only; no validation or phase2 images',
              'training_images':len(rows),'annotation':str(annotation),
              'paired_psr':quantiles([r['paired']['psr'] for r in rows]),
              'wrong_pair_psr':quantiles([r['wrong_pair']['psr'] for r in rows]),
              'paired_displacement_800':quantiles([r['paired_abs_displacement_800'] for r in rows]),
              'synthetic_error_pixels':quantiles([r['error_pixels'] for r in synthetic]),
              'synthetic_exact_fraction':float(np.mean([r['error_pixels']==0 for r in synthetic])),
              'paired_shift_gain':quantiles([r['paired']['edge_correlation_shifted']-r['paired']['edge_correlation_identity'] for r in rows]),
              'wrong_pair_shift_gain':quantiles([r['wrong_pair']['edge_correlation_shifted']-r['wrong_pair']['edge_correlation_identity'] for r in rows]),
              'elapsed_seconds':time.time()-started,
              'limitations':['Phase-correlation peaks are not calibrated cross-spectral matches.',
                             'Synthetic success verifies estimator sign and mechanics, not RGB-IR validity.',
                             'Wrong pairs may share a scene; no registration ground truth exists here.',
                             'Global translation cannot describe parallax, scale or local distortion.',
                             'No labels, training, inference or running jobs modified.']}
    report['paired_psr_exceeds_wrong_fraction'] = float(np.mean([
        r['paired']['psr'] > r['wrong_pair']['psr'] for r in rows]))
    report['descriptive_strata'] = {}
    for cutoff in [10,20]:
        subset = [r for r in rows if r['paired']['psr'] >= cutoff]
        report['descriptive_strata'][str(cutoff)] = {
            'images':len(subset),
            'displacement_800_p50_p90':list(map(float,np.quantile(
                [r['paired_abs_displacement_800'] for r in subset],[.5,.9]))) if subset else None,
            'zero_translation_images':sum(r['paired']['dx']==r['paired']['dy']==0 for r in subset),
            'shift_reduces_edge_correlation_images':sum(
                r['paired']['edge_correlation_shifted'] < r['paired']['edge_correlation_identity']
                for r in subset),
            'note':'Arbitrary descriptive PSR cutoffs, not calibrated acceptance thresholds.'}
    report['next_action'] = (
        'Do not launch global translation warping based on these estimates. '
        'Inspect local object correspondence and reliability before claiming '
        'misregistration is the bottleneck. Wrong-pair shifts can increase correlation.')
    (args.output/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'synthetic.json').write_text(json.dumps(synthetic,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Descriptive gallery; selection deliberately exposes the strongest estimates.
    ranked = sorted(range(len(rows)),key=lambda i:rows[i]['paired']['psr'],reverse=True)[:6]
    gallery = Image.new('RGB',(960,len(ranked)*294),'white')
    draw = ImageDraw.Draw(gallery)
    for j,i in enumerate(ranked):
        for col,view in enumerate(pairs[i][1]):gallery.paste(view,(col*480,j*294+24))
        r=rows[i];p=r['paired']
        draw.text((4,j*294+4),f"TRAIN {r['filename']} | RGB left / IR right | estimated IR shift ({p['dx']},{p['dy']}) PSR {p['psr']:.1f}; UNVERIFIED",fill='black')
    gallery.save(args.output/'strongest_estimates.jpg')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
