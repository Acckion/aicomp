"""Prepare pinned public DEIMv2 source/weights, or verify an offline installation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REV='1d2ca42171570c713e78fc6a766ec5104b7f4724'
WEIGHTS={
    'L':'c0d0f45fa62b785126db5b40f5530866fecaab2d340cea812d2ba5f0f5646163',
    'X':'c15e1860118e6841f8f803219507dbfb630c2aa0164ab723c339b94e554bae7f',
}


def main(offline):
    source=ROOT/'experiments/model_sources/DEIMv2'
    if not source.exists():
        if offline:raise RuntimeError('Missing offline DEIMv2 source')
        source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','https://github.com/Intellindust-AI-Lab/DEIMv2.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',REV],check=True)
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert actual==REV,'Unexpected DEIMv2 source revision; refusing implicit update'
    directory=ROOT/'checkpoints/deimv2';directory.mkdir(parents=True,exist_ok=True)
    for size,expected in WEIGHTS.items():
        path=directory/f'{size}_model.safetensors'
        if not path.exists():
            if offline:raise RuntimeError('Missing offline weight: '+str(path))
            tmp=path.with_suffix('.part')
            urllib.request.urlretrieve(f'https://huggingface.co/Intellindust/DEIMv2_DINOv3_{size}_COCO/resolve/main/model.safetensors',tmp)
            tmp.replace(path)
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,f'Pretrained content mismatch: {size}; do not silently use changed weights'
    (directory/'provenance.json').write_text(json.dumps({'source_revision':REV,'source':'https://github.com/Intellindust-AI-Lab/DEIMv2','weight_sha256':WEIGHTS,'weights_source':'Official Intellindust DEIMv2 DINOv3 L/X COCO Hugging Face repositories','training_data':'Official AICOMP train only; no external training images'},indent=2))
    print('Pinned source and L/X pretrained weights verified; local training can run offline.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');a=parser.parse_args();main(a.offline)
