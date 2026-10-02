"""Own worker entry, including actual successful optimizer-step audit."""
import json,sys,time
from pathlib import Path
import torch
import train_mechanisms
_original=torch.optim.AdamW.step

def audited_step(self,*args,**kwargs):
    checks=[torch.isfinite(p.grad).all() for group in self.param_groups for p in group['params'] if p.grad is not None]
    if checks and not torch.stack(checks).all():
        raise RuntimeError('Non-finite gradient reached actual AdamW step')
    result=_original(self,*args,**kwargs)
    count=getattr(self,'_vf_actual_updates',0)+1;self._vf_actual_updates=count
    if count<=3 or count%100==0:
        cfg=Path(sys.argv[sys.argv.index('--config')+1]);out=Path(__file__).resolve().parents[1]/'runs'/cfg.stem
        if '--smoke' not in sys.argv:
            p=out/'actual_optimizer_steps.json';t=p.with_suffix('.tmp')
            t.write_text(json.dumps({'actual_finite_optimizer_steps':count,'time':time.time()},indent=2));t.replace(p)
    return result
if __name__=='__main__':
    torch.optim.AdamW.step=audited_step

    from src.core import YAMLConfig
    original_property=YAMLConfig.optimizer.fget
    def audited_optimizer(cfg):
        optimizer=original_property(cfg)
        names={id(p):n for n,p in cfg.model.named_parameters()}
        checked=[]
        for group in optimizer.param_groups:
            adapter=[names[id(p)] for p in group['params'] if names[id(p)].startswith('vf_adapter.')]
            if adapter:
                assert group['lr']==0.0003, (adapter, group['lr'])
                checked.extend(adapter)
                print('VF_ADAPTER_OPTIMIZER', group['lr'], adapter, flush=True)
        assert len(checked)==len(list(cfg.model.vf_adapter.parameters()))
        return optimizer
    YAMLConfig.optimizer=property(audited_optimizer)
    train_mechanisms.main()
