"""Fixed epoch4 heldout causal checks for the native-detail branch; no test inference."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native_encoded_ablation_e4'
RUN=ROOT/'runs/scene_native_encoded_delta800'
SSH=['ssh','-i',str(Path.home()/'.ssh/aicomp_servers_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=10','fbohan@222.20.97.104']
REMOTE='/home2/fbohan/AIC_storage/native_encoded/runs/scene_native_encoded_delta800/weights_epoch_004.pth'

def records(path):
    rows={}
    if not path.exists():return rows
    for line in path.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        rows[r['epoch']]=r
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    owner=(OUT/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    def status(stage,**extra):
        p=OUT/'status.tmp';p.write_text(json.dumps({'stage':stage,'controller_pid':os.getpid(),'time':time.time(),**extra},indent=2));p.replace(OUT/'status.json')
    def execute(command,log,env=None):
        nonlocal child
        with log.open('ab') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            status('executing',worker_pid=child.pid,command=command)
            code=child.wait()
        if code:raise RuntimeError(f'Worker failed ({code}); inspect log before retry')
    signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n))
    signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n))
    try:
        plan={'fixed_epoch':4,'modes':['real','zero','shuffle'],'shuffle_seed':20260929,'validation_images':390,'purpose':'Same trained checkpoint: reliance on details and correct spatial correspondence. Not an independent training ablation, not phase2 evidence.','no_test_predictions':True}
        (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
        while 4 not in records(RUN/'metrics.jsonl'):
            state=json.loads((ROOT/'experiments/native_encoded/scene_native_encoded_delta800/status.json').read_text())
            pid=state.get('worker_pid')
            if not pid or not Path(f'/proc/{pid}/cmdline').exists():
                status('training_handle_missing',training_state=state);return
            assert 'scene_native_encoded_delta800.yml' in Path(f'/proc/{pid}/cmdline').read_text()
            status('waiting_epoch4',training_worker_pid=pid)
            time.sleep(60)
        # Copy on the storage host directly, avoiding slow FUSE reads.
        expected=subprocess.check_output([*SSH,'sha256sum',REMOTE],text=True).split()[0]
        checkpoint=Path('/dev/shm/aicomp_native_encoded_e4.pth');temp=checkpoint.with_suffix('.part')
        execute(['scp','-q','-i',str(Path.home()/'.ssh/aicomp_servers_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=10',f'fbohan@222.20.97.104:{REMOTE}',str(temp)],OUT/'copy.log')
        assert hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()==expected
        temp.replace(checkpoint)
        gpu=(ROOT/'experiments/mechanism_trials/gpu4.lock').open('a')
        while True:
            try:
                fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=3072:break
                fcntl.flock(gpu,fcntl.LOCK_UN);status('waiting_memory',free_mib=free)
            except BlockingIOError:status('waiting_project_gpu')
            time.sleep(30)
        rows={}
        for mode in plan['modes']:
            folder=OUT/mode
            command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_variants.py'),'--checkpoint',str(checkpoint),'--config',str(ROOT/'configs/scene_native_encoded_delta800.yml'),'--annotations',str(ROOT/'data/annotations/scene_val.json'),'--image-root',str(ROOT/'data/train'),'--size','800','--amp','--native-top100','--require-ema','--single-method','--native-detail-mode',mode,'--gpu-memory-limit-gib','2','--output',str(folder)]
            execute(command,OUT/f'{mode}.log',{**os.environ,'CUDA_VISIBLE_DEVICES':'4','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            result=json.loads((folder/'results.json').read_text());assert result['images']==390
            identity=json.loads((folder/'cache_identity.json').read_text());assert identity['checkpoint_sha256']==expected and identity['native_detail_mode']==mode
            rows[mode]=result['results'][0]
        differences={mode:{'map_delta':rows['real']['map']-rows[mode]['map'],'per_class_delta':{c:rows['real']['per_class'][c]-rows[mode]['per_class'][c] for c in rows['real']['per_class']}} for mode in ['zero','shuffle']}
        (OUT/'report.json').write_text(json.dumps({**plan,'checkpoint_sha256':expected,'results':rows,'real_minus_ablation':differences,'limitations':'Ablation diagnoses use of evidence; paired continuation control is still required to establish benefit. Does not establish online57.'},indent=2))
        status('complete',report=str(OUT/'report.json'))
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            except ProcessLookupError:pass

if __name__=='__main__':main()
