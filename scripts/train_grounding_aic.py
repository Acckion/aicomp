"""Isolated official MMDetection GroundingDINO supervised AIC training."""
import os,sys,json,time,math,copy,argparse,hashlib,types,fcntl
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/grounding';OUT.mkdir(exist_ok=True)
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmdet.utils import register_all_modules
register_all_modules()
from mmdet.registry import METRICS,HOOKS
from mmdet.evaluation.metrics import CocoMetric
from pycocotools.cocoeval import COCOeval

def atom(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)

@METRICS.register_module()
class GroundingCocoMetric(CocoMetric):
 def compute_metrics(self,results):
  scores=super().compute_metrics(results)
  ce=COCOeval(self._coco_api,self._coco_api.loadRes(self.outfile_prefix+'.bbox.json'),'bbox');ce.params.imgIds=self.img_ids;ce.params.catIds=self.cat_ids;ce.params.maxDets=[1,10,100];ce.evaluate();ce.accumulate();ce.summarize();q=ce.eval['precision'];a=q[np.isclose(ce.params.iouThrs,.9),:,:,0,-1];scores['bbox_AP90']=float(a[a>=0].mean());self.last_coco_stats=ce.stats.tolist();return scores

@HOOKS.register_module()
class GroundingProgress(Hook):
 priority='LOW'
 def before_train(self,runner):
  self.start=time.time();self.updates=0;self.previous_count=0
  # Count optimizer.step, rather than treating accumulating iterations as updates.
  opt=runner.optim_wrapper.optimizer;old=opt.step
  def counted(*a,**kw):
   assert all(torch.isfinite(p.grad).all() for group in opt.param_groups for p in group['params'] if p.grad is not None), 'Non-finite gradients before optimizer update'
   r=old(*a,**kw);self.updates+=1;return r
  opt.step=counted
  atom(OUT/'status.json',{'state':'training','stage':'training','run':'grounding1600','pid':os.getpid(),'gpu':0,'epochs':12,'optimizer_updates':0,'time':time.time()})
 def before_train_epoch(self,runner):
  self.loss_sum=0.;self.loss_count=0
 def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
  loss=float(outputs['loss']);self.loss_sum+=loss;self.loss_count+=1;assert math.isfinite(loss),f'Non-finite training loss: {loss}'
  if (runner.iter+1)%8==0:
   st={'state':'training','stage':'training','run':'grounding1600','pid':os.getpid(),'gpu':0,'epoch':runner.epoch+1,'iteration':runner.iter+1,'batch_in_epoch':batch_idx+1,'batches_per_epoch':len(runner.train_dataloader),'epochs':12,'optimizer_updates':self.updates,'loss':loss,'train_loss_mean':self.loss_sum/max(1,self.loss_count),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'time':time.time()};atom(OUT/'status.json',st);print('GROUNDING_PROGRESS '+json.dumps(st),flush=True)
 def after_val_epoch(self,runner,metrics=None):
  m=runner.val_evaluator.metrics[0];v={'coco_eval_bbox':m.last_coco_stats,'ap90':float(metrics['coco/bbox_AP90'])};row={'epoch':runner.epoch,'validation':v,'metrics':metrics,'train_loss':self.loss_sum/max(1,self.loss_count),'train_batches':self.loss_count,'time':time.time(),'optimizer_updates':self.updates}
  with open(OUT/'metrics.jsonl','a') as f:f.write(json.dumps(row)+'\n')
  atom(OUT/'latest_validation.json',row)
 def after_train(self,runner):
  atom(OUT/'status.json',{'state':'complete','stage':'complete','run':'grounding1600','epochs':12,'optimizer_updates':self.updates,'time':time.time()});(OUT/'COMPLETE').write_text('complete\n')

def main():
 gpu_guard=open(ROOT/'experiments/mechanism_trials/gpu0.lock','a');fcntl.flock(gpu_guard,fcntl.LOCK_EX)
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.backends.cudnn.benchmark=False
 cfg=Config.fromfile(str(ROOT/'configs/grounding1600.py'));cfg.resume=args.resume;cfg.custom_hooks=[dict(type='GroundingProgress')]
 runner=Runner.from_cfg(cfg)
 original_extract=runner.model.extract_feat
 def stable_extract(self,inputs):
  with torch.autocast("cuda",dtype=torch.bfloat16):features=original_extract(inputs)
  return tuple(x.float() for x in features)
 runner.model.extract_feat=types.MethodType(stable_extract,runner.model)
 # MMDet loads pretrained before train; smoke does this explicitly.
 if args.smoke:
  runner.model.init_weights();runner.load_checkpoint(cfg.load_from)
  model=runner.model.cuda();model.train()
  for p in model.language_model.parameters():p.requires_grad_(False)
  # Backprop through checkpointed last backbone stages after frozen preceding features.
  for stage in model.backbone.stages:
   def require(module,a):
    if torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters()):return (a[0].requires_grad_(True),*a[1:])
   stage.register_forward_pre_hook(require)
  state=torch.load(cfg.load_from,map_location='cpu');state=state.get('state_dict',state);own=model.state_dict();missing=[k for k in own if k not in state];mismatch=[k for k,v in own.items() if k in state and state[k].shape!=v.shape];assert not mismatch,mismatch;assert all(k.startswith('bbox_head.cls_branches.') and k.endswith('.log_scale') or k=='dn_query_generator.label_embedding.weight' for k in missing),missing
  atom(OUT/'pretrained_load.json',{'pretrained_keys':len(state),'model_keys':len(own),'missing':missing,'mismatched':mismatch,'extra':[k for k in state if k not in own]})
  # Real max-scale training views, including empty GT, with actual optimizer updates.
  ds=runner.train_dataloader.dataset
  pipe=copy.deepcopy(cfg.train_pipeline);pipe[3]=dict(type='FixScaleResize',scale=(800,1280),keep_ratio=True)
  from mmcv.transforms import Compose
  pipeline=Compose(pipe);indices=sorted(range(len(ds)),key=lambda i:len(ds.get_data_info(i)['instances']),reverse=True)[:3];assert len(indices)==3
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6)
  from mmengine.dataset import pseudo_collate
  losslist=[];scaler=torch.cuda.amp.GradScaler(enabled=False)
  for j,i in enumerate(indices):
   data=ds.get_data_info(i);sample=pipeline(data)
   if j==1:
    from mmengine.structures import InstanceData
    empty=InstanceData();empty.bboxes=sample['data_samples'].gt_instances.bboxes[:0];empty.labels=sample['data_samples'].gt_instances.labels[:0];sample['data_samples'].gt_instances=empty
   batch=model.data_preprocessor(pseudo_collate([sample,copy.deepcopy(sample)]),training=True);optimizer.zero_grad(set_to_none=True)
   inputs=batch['inputs'];batch['inputs']=torch.nn.functional.pad(inputs,(0,1280-inputs.shape[-1],0,800-inputs.shape[-2]));assert batch['inputs'].shape[-2:]==(800,1280)
   for sample_meta in batch['data_samples']:sample_meta.set_metainfo(dict(batch_input_shape=(800,1280),pad_shape=(800,1280)))
   with torch.autocast('cuda',enabled=False):losses=model(**batch,mode='loss');loss,_=model.parse_losses(losses)
   assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads);assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.backbone.named_parameters() if p.requires_grad), 'No backbone gradients';torch.nn.utils.clip_grad_norm_(model.parameters(),.1);scaler.step(optimizer);scaler.update();losslist.append(float(loss));print('SMOKE_UPDATE',j+1,float(loss),tuple(batch['inputs'].shape),'peakMiB',torch.cuda.max_memory_reserved()/2**20,flush=True)
  atom(OUT/'smoke_passed.json',{'optimizer_updates':3,'losses':losslist,'empty_gt_tested':True,'batch_size':2,'precision':'backbone neck bf16; encoder decoder language fp32','maximum_size':[800,1280],'input_padding_stressed':True,'selected_gt_counts':[len(ds.get_data_info(i)['instances']) for i in indices],'backbone_nonzero_gradients':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'time':time.time()});return
 # Freeze language explicitly, retain visual adaptation and multimodal interaction.
 for p in runner.model.language_model.parameters():p.requires_grad_(False)
 for stage in runner.model.backbone.stages:
  def require(module,a):
   if torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters()):return (a[0].requires_grad_(True),*a[1:])
  stage.register_forward_pre_hook(require)
 runner.train()
if __name__=='__main__':main()
