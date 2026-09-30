#!/usr/bin/env bash
set -euo pipefail
cd /home/fbohan/AIC
remote_target=fbohan@222.20.97.104
remote_shell='ssh -i /home/fbohan/.ssh/aicomp_servers_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes'
mkdir -p backups
if [ ! -f backups/AICOMP-env.tar.gz ]; then
 /home/fbohan/miniconda3/bin/conda-pack -p /home/fbohan/miniconda3/envs/AICOMP -o backups/AICOMP-env.tar.gz --compress-level 1
fi
rsync -a -e "$remote_shell" backups/AICOMP-env.tar.gz "$remote_target:/home2/fbohan/"
rsync -a -e "$remote_shell" D-FINE scripts configs data/annotations "$remote_target:/home2/fbohan/AIC/"
$remote_shell "$remote_target" 'mkdir -p /home2/fbohan/AIC/data /home2/fbohan/AIC/runs/rgb1600; if [ -d /home2/fbohan/AIC/annotations ]; then mv /home2/fbohan/AIC/annotations /home2/fbohan/AIC/data/; fi'
rsync -a -e "$remote_shell" data/train "$remote_target:/home2/fbohan/AIC/data/"
rsync -a -e "$remote_shell" runs/rgb1600/weights_epoch_089.pth "$remote_target:/home2/fbohan/AIC/runs/rgb1600/"
$remote_shell "$remote_target" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p /home2/fbohan/miniconda3/envs/AICOMP
if [ ! -f /home2/fbohan/miniconda3/envs/AICOMP/.aicomp_unpacked ]; then
 tar -xzf /home2/fbohan/AICOMP-env.tar.gz -C /home2/fbohan/miniconda3/envs/AICOMP
 /home2/fbohan/miniconda3/envs/AICOMP/bin/python /home2/fbohan/miniconda3/envs/AICOMP/bin/conda-unpack
 touch /home2/fbohan/miniconda3/envs/AICOMP/.aicomp_unpacked
fi
cd /home2/fbohan/AIC
/home2/fbohan/miniconda3/envs/AICOMP/bin/python - <<'PY'
from pathlib import Path
for folder,pattern in [('scripts','*.py'),('configs','*.yml')]:
 for p in Path(folder).glob(pattern):
  p.write_text(p.read_text().replace('/home/fbohan/AIC','/home2/fbohan/AIC').replace('/home/fbohan/miniconda3','/home2/fbohan/miniconda3'))
PY
export LD_LIBRARY_PATH=/home2/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib
/home2/fbohan/miniconda3/envs/AICOMP/bin/python - <<'PY'
import torch,json
from pathlib import Path
assert torch.cuda.is_available()
p=Path('data/annotations/val400.json')
a=json.loads(p.read_text())
assert len(a['images'])==400
assert all((Path('data/train')/x['file_name']).exists() for x in a['images'])
print('Remote CUDA and validation data ready:',torch.cuda.device_count(),'GPUs')
PY
touch REMOTE_READY
REMOTE
printf 'GPU6 environment and training inputs prepared. Queue launch remains separate to avoid duplicate runs.\n'
