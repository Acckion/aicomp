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
    count=getattr(self,'_ir_content_actual_updates',0)+1;self._ir_content_actual_updates=count
    if count<=3 or count%100==0:
        cfg=Path(sys.argv[sys.argv.index('--config')+1]);out=Path(__file__).resolve().parents[1]/'runs'/cfg.stem
        if '--smoke' not in sys.argv:
            p=out/'actual_optimizer_steps.json';t=p.with_suffix('.tmp')
            t.write_text(json.dumps({'actual_finite_optimizer_steps':count,'time':time.time()},indent=2));t.replace(p)
    return result
if __name__=='__main__':
    torch.optim.AdamW.step=audited_step
    train_mechanisms.main()
