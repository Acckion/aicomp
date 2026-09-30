"""Reject datasets, model artifacts and large files in Git's staged snapshot."""
import subprocess
from pathlib import PurePosixPath

FORBIDDEN_ROOTS={'data','runs','checkpoints','backups','logs','monitoring','miniconda3','.ssh','.venv'}
FORBIDDEN_SUFFIXES={'.pth','.pt','.ckpt','.safetensors','.onnx','.engine','.pkl','.pickle','.zip','.tar','.gz','.tgz','.rar','.7z','.npy','.npz','.png','.jpg','.jpeg','.tif','.tiff','.bmp','.mp4','.avi','.pem','.key'}
LIMIT=5*1024*1024

def main():
    paths=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z']).decode().split('\0')
    errors=[]
    for name in filter(None,paths):
        path=PurePosixPath(name)
        forbidden=(path.parts[0] in FORBIDDEN_ROOTS or path.parts[0].startswith('初赛数据集') or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env') or (path.parts[0]=='experiments' and name!='experiments/DECISIONS.md'))
        if forbidden:errors.append(f'{name}: data, output, model, archive or credential file')
        size=int(subprocess.check_output(['git','cat-file','-s',':'+name]))
        if size>LIMIT:errors.append(f'{name}: {size:,} bytes exceeds 5 MiB limit')
    if errors:
        print('Commit rejected. Keep these files outside version control:\n'+'\n'.join(errors))
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
