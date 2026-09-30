"""Explicitly migrate a three-level EMA checkpoint to four-level P2 attention."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'D-FINE'))
import torch
from src.core import YAMLConfig
import p2_encoder


def migrate(current,source,heads=8,old_points=12,added_points=3):
 out={};notes={'added_encoder':[],'expanded_attention':[],'shifted_projection':[]}
 for key,target in current.items():
  if key.startswith(('encoder.p2_detail.','encoder.p2_gate')):
   out[key]=target;notes['added_encoder'].append(key);continue
  if key in ('decoder.anchors','decoder.valid_mask') or key.endswith('.num_points_scale'):
   out[key]=target;continue
  source_key=key
  if key.startswith('decoder.input_proj.'):
   parts=key.split('.');idx=int(parts[2]);parts[2]=str(max(0,idx-1));source_key='.'.join(parts)
   notes['shifted_projection'].append({'destination':key,'source':source_key})
  if source_key not in source:raise ValueError(f'Unmapped pretrained tensor: {source_key}')
  value=source[source_key]
  expanded=any('.'+part+'.' in key for part in ['sampling_offsets','attention_weights'])
  if expanded and target.shape!=value.shape:
   coords=2 if '.sampling_offsets.' in key else 1
   tail=tuple(value.shape[1:])
   assert value.shape[0]==heads*old_points*coords,(key,value.shape)
   assert target.shape[0]==heads*(old_points+added_points)*coords,(key,target.shape)
   new=target.clone().reshape(heads,old_points+added_points,coords,*tail)
   old=value.reshape(heads,old_points,coords,*tail)
   new[:,added_points:]=old
   if key.endswith('attention_weights.bias'):new[:,:added_points]=-2
   out[key]=new.reshape_as(target);notes['expanded_attention'].append(key)
  else:
   if value.shape!=target.shape:raise ValueError(f'Unexpected shape mismatch {key}: {value.shape}, {target.shape}')
   out[key]=value.clone()
 return out,notes


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--config',default=str(ROOT/'configs/p2_640.yml'));ap.add_argument('--output',required=True);a=ap.parse_args()
 torch.manual_seed(20260929);cfg=YAMLConfig(a.config);model=cfg.model
 state=torch.load(a.source,map_location='cpu',weights_only=False);source=state['ema']['module'] if 'ema' in state else state['model']
 weights,notes=migrate(model.state_dict(),source)
 model.load_state_dict(weights,strict=True)
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'model':weights,'p2_initialization':{'source':a.source,'config':a.config,'notes':notes}},path)
 path.with_suffix('.json').write_text(json.dumps(notes,indent=2));print('P2 checkpoint initialized',path,'expanded attention tensors',len(notes['expanded_attention']))
