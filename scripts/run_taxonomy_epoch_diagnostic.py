"""Matched epoch evaluation and TIDE localization diagnostics on the heldout fold."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,required=True);parser.add_argument('--gpu',type=int,default=4)
    args=parser.parse_args();assert args.epoch>0
    out=ROOT/f'experiments/taxonomy_matched_e{args.epoch}';out.mkdir(parents=True,exist_ok=True)
    owner=(out/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    key=str(Path.home()/'.ssh/aicomp_servers_ed25519');host='fbohan@222.20.97.217'
    ssh=['ssh','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=10',host]
    def status(stage,**extra):
        p=out/'status.tmp';p.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),'epoch':args.epoch,'time':time.time(),**extra},indent=2));p.replace(out/'status.json')
    def execute(command,log,env=None):
        nonlocal child
        with log.open('ab') as f:
            child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            status('executing',worker_pid=child.pid,command=command)
            code=child.wait()
        if code:raise RuntimeError(f'Worker failed with {code}: {log}; no blind restart')
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n));signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    try:
        provenance={};weights={}
        for method in ['pool','reset']:
            remote=f'/home/fbohan/AIC/runs/scene_obj365_{method}800/weights_epoch_{args.epoch:03d}.pth'
            sha=subprocess.check_output([*ssh,'sha256sum',remote],text=True).split()[0]
            checkpoint=Path(f'/dev/shm/aicomp_taxonomy_{method}_e{args.epoch}.pth');temp=checkpoint.with_suffix('.part')
            execute(['scp','-q','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=10',f'{host}:{remote}',str(temp)],out/f'{method}_copy.log')
            assert hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()==sha
            temp.replace(checkpoint);weights[method]=checkpoint;provenance[method]={'sha256':sha,'source':f'{host}:{remote}'}
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        for method in ['pool','reset']:
            gpu=(ROOT/f'experiments/mechanism_trials/gpu{args.gpu}.lock').open('a')
            while True:
                try:
                    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free>=3072:break
                    fcntl.flock(gpu,fcntl.LOCK_UN);status('waiting_memory',method=method,free_mib=free)
                except BlockingIOError:status('waiting_gpu',method=method)
                time.sleep(30)
            command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(weights[method]),'--config',str(ROOT/f'configs/scene_obj365_{method}800.yml'),'--annotations',str(ROOT/'data/annotations/scene_val.json'),'--image-root',str(ROOT/'data/train'),'--size','800','--amp','--native-top100','--require-ema','--single-method','--gpu-memory-limit-gib','2','--output',str(out/method)]
            execute(command,out/f'{method}_eval.log',{**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            gpu.close()
            result=json.loads((out/method/'results.json').read_text());assert result['images']==390 and not result['limited_subset']
            identity=json.loads((out/method/'cache_identity.json').read_text());assert identity['checkpoint_sha256']==provenance[method]['sha256']
        for method in ['pool','reset']:
            execute([sys.executable,'-u',str(ROOT/'scripts/tide_diagnostics.py'),'--predictions',str(out/method/'raw_predictions.json'),'--annotations',str(ROOT/'data/annotations/scene_val.json'),'--source',f'{method}_e{args.epoch}','--output',str(out/f'{method}_tide')],out/f'{method}_tide.log',{**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        rows={m:json.loads((out/m/'results.json').read_text())['results'][0] for m in ['pool','reset']}
        tides={m:json.loads((out/f'{m}_tide/report.json').read_text())['mean_idealized_ap_gain'] for m in rows}
        delta={'map':rows['pool']['map']-rows['reset']['map'],'AP50':rows['pool']['stats'][1]-rows['reset']['stats'][1],'AP75':rows['pool']['stats'][2]-rows['reset']['stats'][2],'small_AP':rows['pool']['stats'][3]-rows['reset']['stats'][3],'per_class':{c:rows['pool']['per_class'][c]-rows['reset']['per_class'][c] for c in rows['pool']['per_class']}}
        report={'epoch':args.epoch,'images':390,'checkpoint_provenance':provenance,'results':rows,'pool_minus_reset':delta,'tide_idealized_nonadditive_gains':tides,'limitations':'One matched epoch and one training seed. Tricycle has5GT. TIDE interventions are not attainable improvements. Does not establish phase2 score or online gain.'}
        (out/'report.json').write_text(json.dumps(report,indent=2));status('complete',report=str(out/'report.json'))
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            except ProcessLookupError:pass

if __name__=='__main__':main()
