"""Mirror immutable RAM checkpoint inodes to durable SSH storage, with SHA checks.

The training writer atomically replaces files. A hard link pins each completed
inode while transfer runs; no partial .tmp checkpoint is copied or published.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--remote-dir',required=True)
    parser.add_argument('--host',default='fbohan@222.20.97.104')
    parser.add_argument('--key',type=Path,default=Path.home()/'.ssh/aicomp_servers_ed25519')
    parser.add_argument('--status-dir',type=Path,required=True)
    args=parser.parse_args()
    assert args.source.resolve().is_relative_to(Path('/dev/shm'))
    assert args.remote_dir.startswith('/home2/fbohan/AIC_storage/')
    out=args.status_dir;out.mkdir(parents=True,exist_ok=True)
    owner=(out/'mirror.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshots=args.source/'.mirror_snapshots';snapshots.mkdir(exist_ok=True)
    ssh=['ssh','-i',str(args.key),'-o','BatchMode=yes','-o','ConnectTimeout=10',args.host]
    ledger={};ledger_file=out/'ledger.json'
    if ledger_file.exists():ledger=json.loads(ledger_file.read_text())
    def save_json(path,data):
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
    def status(stage,**extra):
        save_json(out/'status.json',{'stage':stage,'pid':os.getpid(),'time':time.time(),**extra})
    while True:
        try:
            files=sorted(args.source.glob('*.pth'),key=lambda p:(p.name!='last.pth',p.name))
            files += [p for p in args.source.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.yml')]
            complete=args.source/'COMPLETE'
            if complete.exists():files.append(complete)
            for path in files:
                stat=path.stat();signature=[stat.st_ino,stat.st_size,stat.st_mtime_ns]
                if ledger.get(path.name,{}).get('signature')==signature:continue
                assert path.name.replace('_','').replace('.','').isalnum()
                snap=snapshots/f'{os.getpid()}.{path.name}'
                if snap.exists():snap.unlink()
                if path.suffix=='.pth':os.link(path,snap)
                else:
                    data=path.read_bytes()
                    if path.suffix=='.jsonl' and data and not data.endswith(b'\n'):
                        data=data[:data.rfind(b'\n')+1]
                    if path.suffix=='.json':json.loads(data)
                    if path.suffix=='.jsonl':
                        for line in data.splitlines():json.loads(line)
                    snap.write_bytes(data)
                digest=hashlib.file_digest(snap.open('rb'),'sha256').hexdigest()
                temporary=f'{args.remote_dir}/.{path.name}.mirror_{os.getpid()}.part'
                final=f'{args.remote_dir}/{path.name}'
                status('copying',file=path.name,bytes=snap.stat().st_size,sha256=digest)
                started=time.monotonic()
                subprocess.run(['scp','-q','-i',str(args.key),'-o','BatchMode=yes','-o','ConnectTimeout=10',
                    str(snap),f'{args.host}:{temporary}'],check=True)
                program='import hashlib,os,sys; p,q,w=sys.argv[1:]; f=open(p,"rb"); h=hashlib.file_digest(f,"sha256").hexdigest(); f.close(); assert h==w,(h,w); os.replace(p,q); print(h)'
                command=' '.join(map(shlex.quote,['/home2/fbohan/miniconda3/envs/AICOMP/bin/python','-c',program,temporary,final,digest]))
                verified=subprocess.check_output([*ssh,command],text=True).strip();assert verified==digest
                ledger[path.name]={'signature':signature,'sha256':digest,'bytes':snap.stat().st_size,
                    'verified_at':time.time(),'copy_seconds':time.monotonic()-started}
                save_json(ledger_file,ledger);snap.unlink()
            status('synced',files=len(ledger),training_complete=complete.exists())
            if complete.exists():
                final_files=[p for p in args.source.iterdir() if p.is_file() and
                    (p.suffix in ('.pth','.json','.jsonl','.yml') or p.name=='COMPLETE')]
                pending=False
                for path in final_files:
                    stat=path.stat()
                    if ledger.get(path.name,{}).get('signature')!=[stat.st_ino,stat.st_size,stat.st_mtime_ns]:
                        pending=True;break
                if not pending:return
        except Exception:
            status('backup_retry',traceback=traceback.format_exc())
        time.sleep(15)


if __name__=='__main__':main()
