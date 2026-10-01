"""Verify the in-flight runtime archive using storage-server CPU only."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--writer-pid',type=int,required=True);args=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];STORE=ROOT/'checkpoints/gpu6_storage/codino';PID=args.writer_pid
while Path(f'/proc/{PID}').exists():time.sleep(10)
ssh=['ssh','-i','/home/fbohan/.ssh/aicomp_servers_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','fbohan@222.20.97.104']
archive='/home2/fbohan/AIC_storage/codino/env/runtime_py311_torch20_cu118.tar'
subprocess.run(ssh+['tar','-tf',archive],stdout=subprocess.DEVNULL,check=True)
sha=subprocess.check_output(ssh+['sha256sum',archive],text=True).split()[0]
manifest={'sha256':sha,'bytes':(STORE/'env/runtime_py311_torch20_cu118.tar').stat().st_size,'verified_at':time.time(),'runtime':'RAM runtime backed up to authorized GPU6 storage; no GPU6 compute'}
(STORE/'env/runtime_backup.COMPLETE').write_text(json.dumps(manifest,indent=2))
(ROOT/'experiments/codino/runtime_backup.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
