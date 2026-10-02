"""Validate copied official training inputs and relocated Miniconda environment."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import zipfile
import torch

ROOT = Path(__file__).resolve().parents[1]
assert socket.gethostname().lower() == 'gpu8'
manifest = json.loads((ROOT / 'migration/input_manifest.json').read_text())
for relative, expected in {**manifest['files'], **manifest['images']}.items():
    with (ROOT / relative).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == expected, relative
train = json.loads((ROOT / 'data/annotations/train1600.json').read_text())
val = json.loads((ROOT / 'data/annotations/val400.json').read_text())
assert len(train['images']) == 1600 and len(val['images']) == 400
assert set(i['id'] for i in train['images']).isdisjoint(i['id'] for i in val['images'])
assert set(i['file_name'] for i in train['images']).isdisjoint(i['file_name'] for i in val['images'])
with zipfile.ZipFile(ROOT / 'migration/AICOMP_IR_2000.zip') as archive:
    names = {Path(i.filename).name for i in archive.infolist() if not i.is_dir()}
    assert len(names) == 2000
    assert all(Path(i['file_name']).name in names for i in train['images'] + val['images'])
assert torch.cuda.is_available()
assert torch.__version__.startswith('2.5')
assert sys.executable == '/home/fbohan/miniconda3/envs/AICOMP/bin/python'
torch.cuda.set_per_process_memory_fraction(.05)
x = torch.ones(16, 16, device='cuda')
assert torch.isfinite(x @ x).all()
report = {'stage': 'passed', 'server': socket.gethostname(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
          'train': 1600, 'val': 400, 'disjoint': True, 'rgb_sha256_verified': len(manifest['images']),
          'archive_environment_weights_sha256_verified': True}
(ROOT / 'migration/validation.json').write_text(json.dumps(report, indent=2))
(ROOT / 'migration/VALIDATED').write_text('passed\n')
print(json.dumps(report), flush=True)
