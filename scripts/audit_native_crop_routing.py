"""Heldout coverage for label-free crop routing; no AP or test predictions."""
import fcntl,hashlib,importlib,json,os,sys
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'D-FINE'),str(ROOT/'scripts')]
from src.core import YAMLConfig
from native_crop_routing import encoder_candidates,select_windows,grid_windows

@torch.inference_mode()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
    out=ROOT/'experiments/native_crop_routing_e30';out.mkdir(parents=True,exist_ok=True)
    owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=(ROOT/'experiments/mechanism_trials/gpu4.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    config=ROOT/'configs/scene_obj365_pool800.yml';cfg=YAMLConfig(str(config),eval_spatial_size=[800,800])
    for name in cfg.yaml_cfg.get('mechanism_imports',[]):importlib.import_module(name)
    model=cfg.model;checkpoint=Path('/dev/shm/aicomp_taxonomy_pool_e30.pth')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state.get('source')=='EMA'
    weights={k:v for k,v in state['model'].items() if k not in ['decoder.anchors','decoder.valid_mask']}
    missing,extra=model.load_state_dict(weights,strict=False);assert not extra and set(missing)<={'decoder.anchors','decoder.valid_mask'}
    del state,weights
    model.cuda().eval();ann=ROOT/'data/annotations/scene_val.json';data=json.loads(ann.read_text())
    truth={im['id']:[] for im in data['images']}
    for a in data['annotations']:
        if not a.get('iscrowd',0):truth[a['image_id']].append(a)
    crop_rows=json.loads((ROOT/'experiments/native_crop_evidence_val_e30/records.json').read_text())
    helped={r['annotation_id'] for r in crop_rows if r['iou']['0.3']['native_crop']>=.75>r['iou']['0.3']['global']}
    fixed=[(x,y,x+.3,y+.3) for y in [.1,.6] for x in [.1,.6]]
    records=[]
    for index,im in enumerate(data['images']):
        image=Image.open(ROOT/'data/train'/im['file_name']).convert('RGB');w,h=image.size;assert (w,h)==(im['width'],im['height'])
        tensor=TF.to_tensor(image.resize((800,800),Image.Resampling.BILINEAR))[None].cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            features=model.encoder(model.backbone(tensor));boxes,scores=encoder_candidates(model.decoder,features)
        # Selection sees only image-derived candidates. Labels enter below.
        adaptive=select_windows(boxes[0].float(),scores[0].float())
        assert len(adaptive)==4 and len(set(adaptive))==4
        for a in truth[im['id']]:
            x,y,bw,bh=a['bbox']
            if min(bw*800/w,bh*800/h)>=16:continue
            gt=(max(0,x)/w,max(0,y)/h,min(w,x+bw)/w,min(h,y+bh)/h)
            center=((gt[0]+gt[2])/2,(gt[1]+gt[3])/2)
            def coverage(windows):
                return {'center':any(q[0]<=center[0]<=q[2] and q[1]<=center[1]<=q[3] for q in windows),'contained':any(q[0]<=gt[0] and q[1]<=gt[1] and q[2]>=gt[2] and q[3]>=gt[3] for q in windows)}
            records.append({'image_id':im['id'],'annotation_id':a['id'],'class':a['category_id'],'crop_helped75':a['id'] in helped,'adaptive':coverage(adaptive),'uniform4':coverage(fixed),'all16':coverage(grid_windows()),'selected_windows':adaptive})
        (out/'progress.json').write_text(json.dumps({'images':index+1,'small_targets':len(records),'worker_pid':os.getpid()}))
    summary={}
    for cohort,rr in [('all',records),('crop_helped75',[r for r in records if r['crop_helped75']])]:
        summary[cohort]={'targets':len(rr),'coverage':{strategy:{metric:sum(r[strategy][metric] for r in rr) for metric in ['center','contained']} for strategy in ['adaptive','uniform4','all16']}}
    report={'images':len(data['images']),'summary':summary,'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),'annotations_sha256':hashlib.sha256(ann.read_bytes()).hexdigest(),'selector':'top300 encoder candidates, maximum predicted extent<=.15, score>=.05, greedy four of16 .3 windows','peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'limitations':'Coverage only, not AP or proof of localization. all16 is a coverage ceiling with four times the region budget; uniform4 has the same four-region budget. Auxiliary helped cohort is defined using privileged GT-centered crop diagnostics.'}
    (out/'records.json').write_text(json.dumps(records));(out/'report.json').write_text(json.dumps(report,indent=2));(out/'COMPLETE').write_text('ok\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
