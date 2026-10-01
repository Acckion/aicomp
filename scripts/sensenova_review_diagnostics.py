"""After the fixed pilot, test crop targeting on errors and matched controls.

This is an enriched diagnostic set selected using validation labels. It cannot
establish held-out improvement. Ground truth is never included in model inputs.
No model training, score fitting, new boxes or phase2 predictions are performed.
"""
from collections import defaultdict
import fcntl
import io
import json
import os
from pathlib import Path
import random
import string
import subprocess
import sys
import time
import zipfile

from PIL import Image, ImageDraw
import torch

from sensenova_review import ROOT, RAM, ARCHIVE, NAMES, SEED, PROMPT, adjustment
from sensenova_runtime import load_model

PARENT=ROOT/'experiments/sensenova_review'
OUT=PARENT/'crop_diagnostics'
VARIANTS=['crop_only','crop_marked','crop_marked_ir']
TASK=('The first image is an RGB crop around one candidate object. Classify the object inside '
      'the red rectangle when present; otherwise classify the central object. Ignore other '
      'surrounding objects. If a second image is supplied, it is an auxiliary infrared crop '
      'and may be blurred or misaligned. Judge the RGB target; weak infrared evidence should '
      'not overturn clear RGB evidence. '+PROMPT[PROMPT.index('Do not locate'):])

def write(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)

def status(stage,**extra):write('status.json',{'stage':stage,'time':time.time(),**extra})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state=json.loads((PARENT/'controller_status.json').read_text())
        if state['stage']=='complete':break
        if state['stage'] in ['failed','stopped_by_user']:raise RuntimeError('Parent pilot did not complete; diagnostic cancelled')
        status('waiting_for_main_review');time.sleep(30)
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/home/fbohan/miniconda3/envs/AICOMP/lib/python3.11/site-packages/nvidia/nvjitlink/lib'
    with (OUT/'analysis.log').open('ab') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_sensenova_review.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    audit=json.loads((PARENT/'error_analysis/report.json').read_text())
    rows=[json.loads(s) for s in (PARENT/'error_analysis/records_snapshot.jsonl').read_text().splitlines()]
    by=defaultdict(dict)
    for r in rows:by[r['image_id'],r['box_index']][r['variant']]=r
    details=list(__import__('csv').DictReader((PARENT/'error_analysis/candidate_details.csv').open()))
    negatives={(int(r['image_id']),int(r['box_index'])) for r in details if r['positive_iou50']=='False'}
    harmed={(c['image_id'],c['box_index']) for c in audit['harmed_cases']}
    rank=lambda key:-max(by[key][v]['dfine_score']-adjustment(by[key][v],by[key][v]['dfine_score'],.5) for v in by[key])
    bad=sorted(harmed,key=rank)[:24]
    control_tp=sorted((key for key in by if key not in harmed and key not in negatives),key=lambda k:(k[0],k[1]))[:8]
    control_fp=sorted((key for key in negatives if any(adjustment(r,r['dfine_score'],.5)<r['dfine_score'] for r in by[key].values())),key=rank)[:8]
    plan=[{'image_id':i,'box_index':k,'group':group} for group,keys in [('harmed',bad),('positive_control',control_tp),('negative_control',control_fp)] for i,k in keys]
    write('plan.json',plan)
    parent_identity=json.loads((PARENT/'identity.json').read_text())
    identity={'parent_records_sha256':audit['records_sha256'],'parent_model_revision':parent_identity['model_revision'],
              'plan':plan,'variants':VARIANTS,'task_prompt':TASK,
              'scope':'GT-selected diagnostic errors and controls; no held-out or mAP improvement claim'}
    if (OUT/'identity.json').exists():assert json.loads((OUT/'identity.json').read_text())==identity
    else:write('identity.json',identity)
    while True:
        rows_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows_gpu.splitlines()}
        candidates=sorted((g for g in free if free[g]>=9900),key=lambda g:-free[g])
        if len(candidates)>=4:break
        status('waiting_for_gpu_headroom',free_mib=free,total_requests=len(plan)*3);time.sleep(30)
    gpus=candidates[:4];os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
    status('loading_model',gpus=gpus,total_requests=len(plan)*3)
    assert subprocess.check_output(['git','-C',str(RAM/'source'),'rev-parse','HEAD'],text=True).strip()==parent_identity['source_commit']
    assert (RAM/'download.complete').exists()
    model=load_model(RAM/'model',RAM/'source',OUT)
    tokens=[model.tokenizer.encode(c,add_special_tokens=False) for c in string.ascii_uppercase[:14]]
    assert all(len(t)==1 for t in tokens);tokens=[t[0] for t in tokens];state={}
    def constrain(_,inputs,output):
        assert output.shape[0]==1
        masked=torch.full_like(output,-float('inf'))
        if not state:
            logits=output[0].float();subset=logits[tokens];state['probabilities']=subset.softmax(-1).cpu().tolist()
            state['mass']=float(torch.exp(torch.logsumexp(subset,0)-torch.logsumexp(logits,0)));masked[0,tokens]=output[0,tokens]
        else:masked[0,model.new_token_ids['eos_token_id']]=0
        return masked
    hook=model.model.language_model.lm_head.register_forward_hook(constrain)
    record_path=OUT/'records.jsonl';records=[json.loads(s) for s in record_path.read_text().splitlines()] if record_path.exists() else []
    completed={(r['image_id'],r['box_index'],r['variant']) for r in records}
    dataset=json.loads((ROOT/'data/annotations/val400.json').read_text());images={i['id']:i for i in dataset['images']}
    with zipfile.ZipFile(ARCHIVE) as archive,record_path.open('a') as stream:
        members={(Path(p).parts[-2],Path(p).name):p for p in archive.namelist() if len(Path(p).parts)>1 and Path(p).parts[-2] in ['visible','infrared']}
        for c in plan:
            iid,k=c['image_id'],c['box_index'];ref=by[iid,k]['rgb'];name=Path(images[iid]['file_name']).name
            with Image.open(io.BytesIO(archive.read(members['visible',name]))) as im:crop=im.convert('RGB').crop(ref['crop_bounds'])
            with Image.open(io.BytesIO(archive.read(members['infrared',name]))) as im:ir=im.convert('RGB').crop(ref['crop_bounds'])
            marked=crop.copy();left,top,_,_=ref['crop_bounds'];x0,y0,x1,y1=ref['box']
            ImageDraw.Draw(marked).rectangle([x0-left,y0-top,x1-left,y1-top],outline='red',width=1)
            order=ref['option_order'];options=' '.join(f'{string.ascii_uppercase[j]}: {NAMES[cat]}.' for j,cat in enumerate(order))
            prompt=TASK+options+' Answer with exactly one uppercase option letter and nothing else.'
            for variant in VARIANTS:
                if (iid,k,variant) in completed:continue
                state.clear();start=time.monotonic()
                status('diagnosing',image_id=iid,box_index=k,variant=variant,gpus=gpus,completed_requests=len(records),total_requests=len(plan)*3)
                contents=[{'type':'image','value':crop if variant=='crop_only' else marked}]
                if variant=='crop_marked_ir':contents.append({'type':'image','value':ir})
                contents.append({'type':'text','value':prompt})
                with torch.inference_mode():text=model.generate(contents=contents,mode='understanding',noise_seed=SEED+iid+k,max_think_token_n=2)
                letter=str(text).strip();assert letter in string.ascii_uppercase[:14],repr(text)
                choice=order[string.ascii_uppercase.index(letter)];probs=[0.]*14
                for j,cat in enumerate(order):probs[cat]=state['probabilities'][j]
                record={**c,'variant':variant,'choice_id':choice,'choice_name':NAMES[choice],
                        'choice_probabilities':probs,'option_mass':state['mass'],'abstain':bool(choice==13 or state['mass']<.05),
                        'dfine_label':ref['dfine_label'],'dfine_score':ref['dfine_score'],'raw_response':text,
                        'seconds':time.monotonic()-start,'baseline_choices':{v:r['choice_name'] for v,r in by[iid,k].items()},
                        'prompt':prompt,'option_order':order}
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                records.append(record);completed.add((iid,k,variant));print(json.dumps(record,ensure_ascii=False),flush=True)
                torch.cuda.empty_cache()
            summary={}
            for v in VARIANTS:
                rr=[r for r in records if r['variant']==v];hh=[r for r in rr if r['group']=='harmed']
                summary[v]={'completed_regions':len(rr),'harmed_cases_completed':len(hh),
                    'harmed_cases_recognize_original_class':sum(r['choice_id']==r['dfine_label'] for r in hh),
                    'harmed_cases_abstain':sum(r['abstain'] for r in hh),
                    'harmed_cases_still_veto':sum(adjustment(r,r['dfine_score'],.5)<r['dfine_score'] for r in hh)}
            write('summary.json',{'results':summary,'notes':['Enriched diagnostic set, not independent validation or mAP improvement.','Crop-only changes context and task preamble; marked vs unmarked crop changes pixels only.','No training or score fitting.']})
    hook.remove();status('complete',gpus=gpus,completed_requests=len(records),total_requests=len(plan)*3)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
