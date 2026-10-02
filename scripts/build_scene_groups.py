"""Train-pool-only scene grouping by broad-background geometric agreement.

Connected groups are conservative scene candidates, not verified video IDs.
No test images are read. New holdout needs a fresh public-pretrained model;
existing 1600-trained models have seen some images in this new holdout.
"""
from collections import Counter
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/scene_groups'
OUT.mkdir(parents=True, exist_ok=True)


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    cv2.setNumThreads(2)
    source = json.loads((ROOT/'data/annotations/train2000.json').read_text())
    images = source['images']
    assert len(images) == 2000
    descriptors, points, hashes = [], [], []
    orb = cv2.ORB_create(nfeatures=350, scaleFactor=1.2, nlevels=8, edgeThreshold=20)
    began = time.time()
    for index, image in enumerate(images):
        with Image.open(ROOT/'data/train'/image['file_name']) as frame:
            gray = np.asarray(frame.convert('L').resize((512,288),Image.Resampling.BILINEAR))
        kp, desc = orb.detectAndCompute(gray, None)
        descriptors.append(desc)
        points.append(np.asarray([k.pt for k in kp], dtype=np.float32))
        small = cv2.resize(gray,(32,32)).astype(np.float32)
        dct = cv2.dct(small)[:8,:8].flatten()
        hashes.append(dct > np.median(dct[1:]))
        if (index+1) % 100 == 0:
            write(OUT/'status.json', {'stage':'descriptors','images':index+1,'total':2000,'seconds':time.time()-began})
    hashes = np.asarray(hashes, dtype=np.uint8)
    # Integer Hamming matrix with no learned source/test classifier.
    sums = hashes.sum(1).astype(np.int16)
    same_one = hashes.astype(np.int16) @ hashes.astype(np.int16).T
    distance = sums[:,None]+sums[None,:]-2*same_one
    np.fill_diagonal(distance, 100)
    candidates = set()
    for i in range(len(images)):
        for j in np.argsort(distance[i])[:14]:
            if distance[i,j] <= 22:
                candidates.add(tuple(sorted((i,int(j)))))
    parent = list(range(2000))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        a,b = find(i),find(j)
        if a != b:
            parent[b] = a
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    edges = []
    for index,(i,j) in enumerate(sorted(candidates)):
        if descriptors[i] is None or descriptors[j] is None:
            continue
        matches = matcher.knnMatch(descriptors[i],descriptors[j],k=2)
        good = [m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .72*n.distance]
        if len(good) >= 25:
            a = points[i][[m.queryIdx for m in good]]
            b = points[j][[m.trainIdx for m in good]]
            _, valid = cv2.findHomography(a,b,cv2.RANSAC,3.)
            if valid is not None:
                mask = valid[:,0].astype(bool)
                count = int(mask.sum())
                coverage_a = cv2.contourArea(cv2.convexHull(a[mask]))/(512*288) if count>=3 else 0
                coverage_b = cv2.contourArea(cv2.convexHull(b[mask]))/(512*288) if count>=3 else 0
                if count >= 22 and count/len(good) >= .45 and min(coverage_a,coverage_b) >= .18:
                    union(i,j)
                    edges.append({'i':images[i]['id'],'j':images[j]['id'],'inliers':count,'coverage':min(coverage_a,coverage_b)})
        if (index+1) % 500 == 0:
            write(OUT/'status.json',{'stage':'geometric_links','pairs':index+1,'total':len(candidates),'links':len(edges),'seconds':time.time()-began})
    groups = {}
    for i,image in enumerate(images):
        groups.setdefault(find(i),[]).append(image['id'])
    group_list = sorted(groups.values(),key=lambda g:(-len(g),min(g)))
    old_train = {i['id'] for i in json.loads((ROOT/'data/annotations/train1600.json').read_text())['images']}
    old_val = {i['id'] for i in json.loads((ROOT/'data/annotations/val400.json').read_text())['images']}
    overlap = sum(len(set(g)&old_val) for g in group_list if set(g)&old_train)
    annotation_counts = {}
    for ann in source['annotations']:
        annotation_counts.setdefault(ann['image_id'],Counter())[ann['category_id']] += 1
    total = Counter(a['category_id'] for a in source['annotations'])
    target = {k:v*.2 for k,v in total.items()}
    # Candidate-only subset search. Labels stratify official training groups;
    # no model predictions or phase2 information select the partition.
    best = None
    rng = random.Random(20261002)
    for trial in range(500):
        order = group_list.copy()
        rng.shuffle(order)
        chosen, count, cat = [],0,Counter()
        for group in order:
            if count >= 360 and count+len(group)>450:
                continue
            if count+len(group)>500:
                continue
            gc = sum((annotation_counts.get(i,Counter()) for i in group),Counter())
            if any(total[k]-cat[k]-gc[k] < max(5,.4*total[k]) for k in total):
                continue
            chosen.append(group)
            count += len(group)
            cat.update(gc)
            if count >= 390:
                break
        if not 300 <= count <= 500:
            continue
        score = abs(count-400)/400 + sum(abs(cat[k]-target[k])/max(target[k],5) for k in total)/len(total)
        score += 5*sum(cat[k]==0 for k in total)
        if best is None or score<best[0]:
            best = score, chosen, cat
    report = {'groups':group_list,'accepted_edges':edges,'candidate_pairs':len(candidates),
              'group_count':len(group_list),'largest_groups':[len(g) for g in group_list[:20]],
              'old_val_shared_scene_candidates':overlap,'old_val_images':400,
              'note':'Geometric grouping candidates; inspect broad background matches before trusting semantic scene IDs. No test input. Existing mature models cannot independently evaluate the new held-out images.'}
    if best is not None:
        _, chosen, cat = best
        val_ids = {i for g in chosen for i in g}
        train_ids = {i['id'] for i in images}-val_ids
        assert all(set(g)<=train_ids or set(g)<=val_ids for g in group_list)
        for label, ids in [('scene_train',train_ids),('scene_val',val_ids)]:
            subset = {**source,'images':[i for i in images if i['id'] in ids],
                      'annotations':[a for a in source['annotations'] if a['image_id'] in ids]}
            write(ROOT/'data/annotations'/f'{label}.json',subset)
        report.update(new_train_images=len(train_ids),new_val_images=len(val_ids),new_val_counts=dict(cat),all_classes_in_val=all(cat[k]>0 for k in total))
    write(OUT/'report.json',report)
    write(OUT/'status.json',{'stage':'complete','seconds':time.time()-began,'groups':len(group_list)})
    print(json.dumps({k:v for k,v in report.items() if k not in ['groups','accepted_edges']}),flush=True)


if __name__ == '__main__':
    main()
