"""Train-only lightweight correspondence adaptation; CMRW-inspired, not its reproduction.
Known image transforms supervise within-modality correspondence only. Cross-modal
cycles are self-supervised consistency, never asserted to be true IR alignment.
"""
import argparse, json, math, os, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import ir_content_alignment
from src.core import YAMLConfig
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/ir_corr'
class Correspondence(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.rgb=nn.Sequential(nn.Conv2d(dim,128,1),nn.GELU(),nn.Conv2d(128,64,1,bias=False))
        self.ir=nn.Sequential(nn.Conv2d(dim,128,1),nn.GELU(),nn.Conv2d(128,64,1,bias=False))
    def encode(self,x,mode):
        return F.normalize(getattr(self,mode)(x.float()),dim=1,eps=1e-6)

def tokens(x):return x.flatten(2).transpose(1,2)
def affine(device):
    theta=torch.tensor([[[random.uniform(.80,1.02),0,random.uniform(-.2,.2)],[0,random.uniform(.80,1.02),random.uniform(-.2,.2)]]],device=device)
    return theta

def intra(head,original,view,theta,mode):
    a=head.encode(original,mode);b=head.encode(view,mode)
    grid=F.affine_grid(theta,b.shape,align_corners=False)
    # Positive targets sample original feature coordinates through the known
    # transform. Only intra-modal transformation supplies a correspondence GT.
    target=F.grid_sample(a,grid,align_corners=False)
    good=(grid.abs()<.88).all(-1).flatten()
    av=tokens(target)[0][good];bv=tokens(b)[0][good]
    if len(av)<8:return a.sum()*0
    logits=bv@av.T/.10
    labels=torch.arange(len(av),device=a.device)
    loss=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
    with torch.no_grad():
        coords=grid.flatten(1,2)[0][good];pred=coords[logits.argmax(-1)];error=(pred-coords).norm(dim=-1)/2
        head.last_intra_diagnostic={'normalized_endpoint_error':float(error.mean()),'recall_1cell':float((error<1/16).float().mean())}
    return loss

def cross(head,rgb,ir,mask):
    a=tokens(head.encode(rgb,'rgb'))[0];b=tokens(head.encode(ir,'ir'))[0]
    valid=F.interpolate(mask,size=ir.shape[-2:],mode='nearest').flatten()>.5
    b=b[valid]
    if len(b)<8:return a.sum()*0,{'cycle':0.,'entropy':0.,'valid_tokens':int(len(b))}
    sim=a@b.T/.10;p=sim.softmax(-1);q=sim.T.softmax(-1)
    cycle=(p*q.T).sum(-1).clamp_min(1e-8)
    loss=-cycle.log().mean()
    entropy=-(p*p.clamp_min(1e-8).log()).sum(-1).mean()
    return loss,{'cycle':float(cycle.detach().mean()),'entropy':float(entropy.detach()),'valid_tokens':int(len(b))}

def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--epochs',type=int,default=4);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);random.seed(20261002);torch.manual_seed(20261002)
    torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=YAMLConfig(str(ROOT/'configs/ir_corr_pretrain.yml'));ds=cfg.train_dataloader.dataset
    assert len(ds)==1600 and set(ds.ids).isdisjoint(cfg.val_dataloader.dataset.ids)
    # deterministic photometric/geometric paired source, augment images explicitly
    ds.training=False;ds.size=992 if args.preflight else 800
    model=cfg.model
    state=torch.load(ROOT/'runs/ft_aug800/weights_epoch_020.pth',map_location='cpu',weights_only=False)['model']
    for k in ('decoder.anchors','decoder.valid_mask'):state[k]=model.state_dict()[k]
    result=model.load_state_dict(state,strict=False);assert not result.unexpected_keys
    assert all(k.startswith(('ir_backbone.','ir_encoder.','ir_samplers.')) for k in result.missing_keys)
    del state
    model.requires_grad_(False).eval().cuda();model.encoder.eval_spatial_size=None;model.ir_encoder.eval_spatial_size=None
    head=Correspondence(model.encoder.hidden_dim).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    def feat(x,mode):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            branch=(model.backbone,model.encoder) if mode=='rgb' else (model.ir_backbone,model.ir_encoder)
            return F.adaptive_avg_pool2d(branch[1](branch[0](x))[0],(16,16)).detach().float()
    def update(index,stage):
        x,_=ds[index];x=x[None].cuda();r=feat(x[:,:3],'rgb');i=feat(x[:,3:6],'ir');th=affine(x.device)
        g=F.affine_grid(th,x[:,:3].shape,align_corners=False)
        rv=feat(F.grid_sample(x[:,:3],g,align_corners=False),'rgb');iv=feat(F.grid_sample(x[:,3:6],g,align_corners=False),'ir')
        optimizer.zero_grad(set_to_none=True)
        intr=intra(head,r,rv,th,'rgb')+intra(head,i,iv,th,'ir');cyc,diag=cross(head,r,i,x[:,6:7]);loss=intr+(0.05*cyc if stage else 0.)
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.));assert norm>0;optimizer.step()
        return {'loss':float(loss.detach()),'intra_loss':float(intr.detach()),'cross_loss':float(cyc.detach()),'grad_norm':norm,**diag,**head.last_intra_diagnostic},r,i,x[:,6:7]
    if args.preflight:
        records=[]
        for idx in [10,11,12]:
            row,r,i,mask=update(idx,1);records.append(row)
        zero,diag=cross(head,r,i,torch.zeros_like(mask));assert torch.isfinite(zero);assert float(zero)==0
        path=OUT/'preflight_weights.pth';torch.save(head.state_dict(),path);head.load_state_dict(torch.load(path,weights_only=True),strict=True);path.unlink()
        atomic(OUT/'preflight.json',{'stage':'passed','max_size':992,'batch':1,'peak_mib':torch.cuda.max_memory_allocated()/1024**2,'frozen_backbone_no_grad':True,'strict_reload':True,'missing_ir_finite':True,'updates':records});print('PREFLIGHT_PASSED',records,flush=True);return
    def diagnose():
        items=[];errors=[];recalls=[];variances=[]
        head.eval()
        with torch.no_grad():
            th=torch.tensor([[[.85,0,.10],[0,.90,-.10]]],device='cuda')
            for idx in [10,11,12,13]:
                x,_=ds[idx];x=x[None].cuda();r=feat(x[:,:3],'rgb');i=feat(x[:,3:6],'ir')
                grid=F.affine_grid(th,x[:,:3].shape,align_corners=False)
                for mode,orig,image in [('rgb',r,x[:,:3]),('ir',i,x[:,3:6])]:
                    view=feat(F.grid_sample(image,grid,align_corners=False),mode);intra(head,orig,view,th,mode)
                    errors.append(head.last_intra_diagnostic['normalized_endpoint_error']);recalls.append(head.last_intra_diagnostic['recall_1cell'])
                    variances.append(float(tokens(head.encode(orig,mode)).std(dim=1).mean()))
                items.append((r,i,x[:,6:7]))
            paired=[];wrong=[];entropy=[]
            for k,(r,i,mask) in enumerate(items):
                _,d=cross(head,r,i,mask);paired.append(d['cycle']);entropy.append(d['entropy'])
                _,d=cross(head,r,items[(k+1)%len(items)][1],items[(k+1)%len(items)][2]);wrong.append(d['cycle'])
        head.train()
        return {'endpoint_error':sum(errors)/len(errors),'recall_1cell':sum(recalls)/len(recalls),'feature_std':min(variances),'paired_cycle':sum(paired)/len(paired),'wrong_image_cycle':sum(wrong)/len(wrong),'entropy':sum(entropy)/len(entropy)}
    baseline=diagnose();atomic(OUT/'diagnostic_initial.json',baseline)
    total_steps=args.epochs*len(ds);step=0;previous_ir=None;previous_mask=None
    for epoch in range(args.epochs):
        order=list(range(len(ds)));random.shuffle(order);sums={};start=time.time()
        for j,index in enumerate(order):
            lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*step/total_steps)))
            for group in optimizer.param_groups:group['lr']=lr
            row,r,i,mask=update(index,epoch>=1);step+=1
            with torch.no_grad():
                if previous_ir is not None:
                    _,other=cross(head,r,previous_ir,previous_mask);row['wrong_image_cycle']=other['cycle']
                shuffled=i.flatten(2)[:,:,torch.randperm(256,device=i.device)].reshape_as(i)
                _,shuf=cross(head,r,shuffled,mask);row['shuffled_cycle']=shuf['cycle']
                # Cycle ignores positions and therefore spatial shuffle can
                # remain invariant: explicitly disclose this limitation.
                previous_ir=i.detach();previous_mask=mask.detach()
            row.update(epoch=epoch+1,iteration=j+1,steps=step,lr=lr,time=time.time(),elapsed=time.time()-start)
            for k in ['loss','intra_loss','cross_loss','cycle']:sums[k]=sums.get(k,0)+row[k]
            if j%20==0:
                atomic(OUT/'progress.json',row);print(json.dumps(row),flush=True)
        diagnostic=diagnose();atomic(OUT/f'diagnostic_epoch_{epoch+1:03d}.json',diagnostic)
        summary={k:v/len(ds) for k,v in sums.items()};summary.update(epoch=epoch+1,time=time.time(),stage='intramodal' if epoch==0 else 'crossmodal')
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        torch.save({'model':head.state_dict(),'epoch':epoch+1,'train_ids':ds.ids,'architecture':'rgb_ir_384_128_64_no_position','summary':summary},OUT/f'weights_epoch_{epoch+1:03d}.pth')
    final=diagnose();improved=final['recall_1cell']>baseline['recall_1cell']+.01 or final['endpoint_error']<baseline['endpoint_error']*.95
    gate={'stage':'passed' if improved and final['paired_cycle']>=final['wrong_image_cycle']*.95 and final['feature_std']>.001 and final['entropy']>.05 else 'no_evidence','initial':baseline,'final':final,'note':'Known intra transform retrieval; cross cycle consistency is not physical alignment GT.'}
    atomic(OUT/'evidence_gate.json',gate)
    (OUT/'COMPLETE').write_text('completed\n')
if __name__=='__main__':main()
