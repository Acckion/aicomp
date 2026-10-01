"""Restore the isolated Co-DINO runtime after host restart, without changing AICOMP."""
from pathlib import Path
import subprocess,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];RAM=Path('/dev/shm/aicomp_codino');STORE=ROOT/'checkpoints/gpu6_storage/codino'
RAM.mkdir(parents=True,exist_ok=True)
python=RAM/'venv/bin/python'
if not python.exists():
 assert (STORE/'env/runtime_backup.COMPLETE').exists(), 'Runtime backup is not yet verified complete'
 subprocess.run(['tar','-xf',str(STORE/'env/runtime_py311_torch20_cu118.tar'),'-C',str(RAM)],check=True)
weights=RAM/'co_dino_5scale_swin_large_16e_o365tococo.pth'
if not weights.exists():shutil.copyfile(STORE/'weights'/weights.name,weights)
sha=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest()
assert sha=='bc36a67f964099d7e05ddc208dcec9c26dda87d3140382c513f184a34dde65a6',sha
subprocess.run([str(python),'-c','import torch,numpy,cv2,mmcv,mmcv.ops,fairscale,timm; assert torch.__version__=="2.0.0+cu118"; assert numpy.__version__=="1.26.4"; assert cv2.__version__=="4.11.0"; assert mmcv.__version__=="1.7.2"; print(torch.from_numpy(numpy.zeros(2)))'],check=True)
print('Isolated Co-DINO runtime restored and verified.')
