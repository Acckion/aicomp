"""Persist isolated GroundingDINO additions; shared Co-DINO torch runtime is read-only."""
import hashlib,json,os,tarfile,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=Path('/dev/shm/aicomp_grounding');DST=ROOT/'checkpoints/gpu6_storage/grounding';DST.mkdir(parents=True,exist_ok=True)
tmp=SRC/'grounding_runtime.tar.gz';final=DST/'grounding_runtime.tar.gz'
with tarfile.open(tmp,'w:gz',compresslevel=1) as t:
 for name in ['venv','vendor','bert','mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl']:
  t.add(SRC/name,arcname='aicomp_grounding/'+name)
shutil.copyfile(tmp,final)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
a=sha(tmp);b=sha(final);assert a==b
info={'source':str(SRC),'archive':str(final),'sha256':a,'bytes':final.stat().st_size,'shared_dependency':'restore existing /dev/shm/aicomp_codino/venv before grounding; never replace Co-DINO MMCV1','restore':'tar -xzf grounding_runtime.tar.gz -C /dev/shm ; copy grounding_swinb.pth to /dev/shm/aicomp_grounding/grounding_swinb.pth','time':time.time()}
(ROOT/'experiments/grounding/runtime_backup.json').write_text(json.dumps(info,indent=2));(DST/'RUNTIME_COMPLETE').write_text(a+'\n');print(json.dumps(info),flush=True)
