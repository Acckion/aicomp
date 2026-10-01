"""Isolated official Co-DINO transfer, train-only official AIC RGB and held-out validation."""
import argparse, contextlib, gc, json, math, os, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
VENDOR=Path(os.environ.get('CODINO_VENDOR','/dev/shm/aicomp_codino/vendor/Co-DETR'))
sys.path.insert(0,str(VENDOR))
import mmcv
# Isolated vendor's upper bound is widened to 1.7.2; retain truthful runtime version.
assert mmcv.__version__=='1.7.2', mmcv.__version__
from mmcv import Config
from mmcv.parallel import scatter, collate
import projects.models
from mmdet.models import build_detector
from mmdet.datasets import build_dataset,build_dataloader
from pycocotools.cocoeval import COCOeval

OUT=ROOT/'experiments/codino'

def atom(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def configure(cfg,size=None):
    if size:
        for t in cfg.data.train.pipeline:
            if t['type']=='Resize':t['img_scale']=[(int(size*16/9),size)]
    cfg.model.backbone.pretrained=None
    cfg.model.backbone.use_checkpoint=True
    cfg.model.query_head.transformer.encoder.with_cp=6
    return cfg

def freeze(model,partial):
    for n,p in model.backbone.named_parameters():
        p.requires_grad=bool(partial and (n.startswith('layers.3.') or n.startswith('norm3.')))
    model.backbone.eval()

def build(cfg,pretrained,partial=False):
    model=build_detector(cfg.model)
    model.init_weights()
    state=torch.load(pretrained,map_location='cpu');state=state.get('state_dict',state.get('model',state))
    own=model.state_dict();selected={};skip=[]
    # Native COCO class initialization for semantically exact/common labels;
    # animal uses the mean of COCO's ten animal class parameters.
    coco_ids=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
    map_ids={0:[1],1:[9],2:list(range(16,26)),3:[15],4:[13],5:[2],6:[3],7:[37],8:[10]}
    transferred=[]
    for name,t in own.items():
        old=state.get(name)
        if old is not None and old.shape==t.shape:selected[name]=old;continue
        if old is not None and old.ndim==t.ndim and old.shape[1:]==t.shape[1:] and old.shape[0] in (80,81) and t.shape[0] in (12,13):
            new=t.clone()
            for cls,ids in map_ids.items():new[cls]=old[[coco_ids.index(i) for i in ids]].mean(0)
            if t.shape[0]==13 and old.shape[0]==81:new[-1]=old[-1]
            selected[name]=new;transferred.append(name)
        elif old is not None and old.shape[0]==320 and t.shape[0]==48 and old.shape[1:]==t.shape[1:]:
            new=t.clone().reshape(12,4,*t.shape[1:]);old=old.reshape(80,4,*old.shape[1:])
            for cls,ids in map_ids.items():new[cls]=old[[coco_ids.index(i) for i in ids]].mean(0)
            selected[name]=new.reshape_as(t);transferred.append(name)
        else:skip.append(name)
    model.load_state_dict(selected,strict=False)
    atom(OUT/'pretrained_load.json',{'copied_keys':len(selected),'model_keys':len(own),'unmatched':skip,'class_transferred':transferred,'training_only_source':'official COCO/Objects365 detector; no target test data'})
    del state,selected;gc.collect()
    freeze(model,partial)
    # Reentrant checkpoint requires a grad-enabled input when preceding stages are frozen.
    def ensure_stage_input(module,args):
        if torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters()):
            return (args[0].requires_grad_(True),*args[1:])
    model.backbone.layers[3].register_forward_pre_hook(ensure_stage_input)
    # BF16 only for backbone/neck (native RTX3090); legacy deformable attention remains FP32.
    # FP16 overflow on empty frames is avoided without dropping difficult examples.
    def extract(self,img,img_metas=None):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            x=self.backbone(img)
            if self.with_neck:x=self.neck(x)
        return tuple(t.float() for t in x)
    model.extract_feat=types.MethodType(extract,model)
    return model.cuda()

def batch_cuda(batch):return scatter(batch,[0])[0]

def scalar_loss(losses):
    items={}
    for k,v in losses.items():
        if isinstance(v,(list,tuple)):items[k]=sum(x.mean() for x in v)
        elif torch.is_tensor(v):items[k]=v.mean()
    return sum(v for k,v in items.items() if 'loss' in k),items

@torch.no_grad()
def evaluate(model,loader,dataset,run,epoch,limit=None):
    model.eval();results=[];start=time.time()
    for i,data in enumerate(loader):
        data=batch_cuda(data)
        result=model(return_loss=False,rescale=True,**data)
        results.extend(result)
        if i%25==0:print(f'VAL epoch={epoch} image={i+1}/{len(loader)} allocated={torch.cuda.memory_allocated()/2**20:.0f}MiB',flush=True)
        if limit and i+1>=limit:break
    if limit:return {'smoke_predictions':len(results)}
    files,tmp=dataset.format_results(results,str(run/f'predictions_epoch_{epoch:03d}'))
    pred=dataset.coco.loadRes(files['bbox']);ev=COCOeval(dataset.coco,pred,'bbox');ev.params.imgIds=dataset.img_ids;ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    stats=ev.stats.tolist();precision=ev.eval['precision'];valid=precision[8,:,:,0,2];ap90=float(valid[valid>-1].mean())
    return {'coco_eval_bbox':stats,'map':100*stats[0],'ap75':100*stats[2],'aps':100*stats[3],'ap90':100*ap90,'seconds':time.time()-start}

def save(run,model,opt,scaler,epoch,best,args,name):
    path=run/name;tmp=run/(name+'.tmp')
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'args':vars(args),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'random_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp)
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/codino_aic.py'));p.add_argument('--pretrained',required=True);p.add_argument('--epochs',type=int,default=16);p.add_argument('--run',default='codino1600');p.add_argument('--smoke',action='store_true');p.add_argument('--size',type=int);p.add_argument('--resume');p.add_argument('--backbone-mode',choices=['staged','frozen'],default='staged');a=p.parse_args()
    assert torch.cuda.is_bf16_supported(), 'This guarded profile requires native BF16'
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=True;random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002)
    cfg=configure(Config.fromfile(a.config),a.size)
    for imported in ('Path','ROOT'):cfg.pop(imported,None)
    run=ROOT/'runs'/a.run;run.mkdir(parents=True,exist_ok=True)
    cfg.dump(str(run/'effective_config.py'))
    train=build_dataset(cfg.data.train);val=build_dataset(cfg.data.val)
    assert set(train.img_ids).isdisjoint(val.img_ids), 'Training/validation leak'
    assert tuple(train.CLASSES)==tuple(cfg.classes);assert train.cat_ids==list(range(12)),train.cat_ids
    atom(run/'dataset_manifest.json',{'train_images':len(train),'val_images':len(val),'train_ids_disjoint':True,'classes':train.CLASSES,'train_images_path':cfg.data.train.img_prefix,'input':'official normalization/aspect-ratio-preserving resize/pad32','test_used':False})
    model=build(cfg,a.pretrained);model.CLASSES=train.CLASSES
    # Include frozen last-stage parameters in optimizer so unfreezing preserves head states.
    groups=[{'params':[p for n,p in model.named_parameters() if not n.startswith('backbone.')],'lr':1e-4,'initial_lr':1e-4}, {'params':[p for n,p in model.named_parameters() if n.startswith('backbone.layers.3.') or n.startswith('backbone.norm3.')],'lr':1e-5,'initial_lr':1e-5}]
    opt=torch.optim.AdamW(groups,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(enabled=False);start_epoch=0;best=-1.
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu');model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
        if a.backbone_mode=='frozen':
            for parameter in opt.param_groups[1]['params']:opt.state.pop(parameter,None)
        scaler.load_state_dict(saved['scaler']);start_epoch=saved['epoch']+1;best=saved['best'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng']);random.setstate(saved['random_rng']);np.random.set_state(saved['numpy_rng']);del saved;gc.collect()
    trainloader=build_dataloader(train,samples_per_gpu=1,workers_per_gpu=2,num_gpus=1,dist=False,shuffle=True,seed=20261002,persistent_workers=True)
    valloader=build_dataloader(val,samples_per_gpu=1,workers_per_gpu=2,num_gpus=1,dist=False,shuffle=False,persistent_workers=True)
    if a.smoke:
        # Force the largest configured resolution on a dense real frame.
        from collections import Counter
        counts=Counter(x['image_id'] for x in train.coco.dataset['annotations']);dense=max(range(len(train)),key=lambda i:counts[train.img_ids[i]])
        max_size=a.size or 736
        for t in train.pipeline.transforms:
            if t.__class__.__name__=='Resize':t.img_scale=[(int(max_size*16/9),max_size)]
        for partial in (False,True) if a.backbone_mode=='staged' else (False,):
            freeze(model,partial);model.train();model.backbone.eval();opt.zero_grad(set_to_none=True)
            for step in range(2):
                data=batch_cuda(collate([train[dense]],samples_per_gpu=1))
                if step==1:data['gt_bboxes']=[x[:0] for x in data['gt_bboxes']];data['gt_labels']=[x[:0] for x in data['gt_labels']]
                loss,parts=scalar_loss(model(return_loss=True,**data));assert torch.isfinite(loss),parts
                scaler.scale(loss).backward();scaler.unscale_(opt)
                if partial and step==0:
                    gradients=[p.grad for n,p in model.backbone.named_parameters() if n.startswith('layers.3.') and p.requires_grad and p.grad is not None]
                    assert gradients and any(float(g.abs().max())>0 for g in gradients), 'Last Swin stage has no gradient'
                    print(f'SMOKE last-stage nonzero gradients: {len(gradients)} tensors',flush=True)
                norm=torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],.1);assert torch.isfinite(norm), 'Smoke requires actual finite-gradient optimizer step';scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                print(f'SMOKE partial={partial} step={step} shape={list(data["img"].shape)} loss={loss.item():.5f} peak={torch.cuda.max_memory_allocated()/2**20:.1f}MiB',flush=True)
            del data,parts,loss;gc.collect();torch.cuda.empty_cache()
        evaluate(model,valloader,val,run,-1,limit=2)
        atom(OUT/'smoke_passed.json',{'size':max_size,'backbone_mode':a.backbone_mode,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'amp_dtype':'bf16 backbone/neck + fp32 detector heads','time':time.time()});print('SMOKE TEST PASSED',flush=True);return
    for epoch in range(start_epoch,a.epochs):
        partial=(epoch>=2 and a.backbone_mode=='staged');freeze(model,partial);model.train();model.backbone.eval();opt.zero_grad(set_to_none=True);start=time.time();total=0.;updates=0;skipped_updates=0
        for i,data in enumerate(trainloader):
            progress=epoch+i/len(trainloader);factor=min(1.,(progress+1/len(trainloader))/1.)*(.1+.9*.5*(1+math.cos(math.pi*min(progress/a.epochs,1))))
            for g in opt.param_groups:g['lr']=g['initial_lr']*factor
            data=batch_cuda(data);loss,parts=scalar_loss(model(return_loss=True,**data))
            if not torch.isfinite(loss):
                atom(run/'nonfinite_loss.json',{'epoch':epoch+1,'batch':i+1,'components':{k:float(v.detach()) for k,v in parts.items()},'images':[m.get('filename') for m in data['img_metas']]})
                raise RuntimeError(f'Nonfinite loss at epoch={epoch+1} batch={i+1}; see nonfinite_loss.json')
            scaler.scale(loss/8).backward();total+=loss.item()
            if (i+1)%8==0 or i+1==len(trainloader):
                scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],.1)
                before_scale=scaler.get_scale();scaler.step(opt);scaler.update();after_scale=scaler.get_scale();opt.zero_grad(set_to_none=True)
                if after_scale<before_scale:
                    skipped_updates+=1;print(f'AMP overflow skipped epoch={epoch+1} batch={i+1} scale={before_scale}->{after_scale}',flush=True)
                else:
                    assert torch.isfinite(norm), 'Nonfinite gradients without a GradScaler skip'
                    updates+=1
            if i%20==0:
                atom(run/'optimizer_progress.json',{'epoch':epoch+1,'batch':i+1,'batches':len(trainloader),'updates':updates,'amp_skipped_updates':skipped_updates,'amp_scale':scaler.get_scale(),'loss':float(loss),'lr':opt.param_groups[0]['lr'],'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'partial_backbone':partial,'time':time.time()})
                print(f'TRAIN epoch={epoch+1}/{a.epochs} batch={i+1}/{len(trainloader)} loss={loss.item():.4f} peak={torch.cuda.max_memory_allocated()/2**20:.1f}MiB',flush=True)
            del data,loss,parts
        val_metrics=evaluate(model,valloader,val,run,epoch+1);row={'epoch':epoch+1,'train_loss':total/len(trainloader),'optimizer_updates':updates,'amp_skipped_updates':skipped_updates,'validation':val_metrics,'epoch_seconds':time.time()-start,'partial_backbone':partial}
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        improved=val_metrics['map']>best;best=max(best,val_metrics['map']);save(run,model,opt,scaler,epoch,best,a,'last.pth')
        if improved:save(run,model,opt,scaler,epoch,best,a,'best.pth')
        atom(run/'summary.json',{'epoch':epoch+1,'best_map':best,'latest':row,'time':time.time()})
        print(json.dumps(row),flush=True)
    (run/'COMPLETE').write_text('completed '+str(time.time()))
if __name__=='__main__':main()
