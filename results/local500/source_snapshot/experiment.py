import json
import shutil
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, set_seed
from .common import begin_run, read_json, write_json, sha256, snapshot_model, checksums
from .data import image_path
from .evaluation import evaluate

SETTINGS={'greedy':{'beams':1,'image':'original'},'beam3':{'beams':3,'image':'original'},
          'blank_image':{'beams':1,'image':'uniform RGB 127; same original dimensions'}}


def load_model(device='auto'):
    device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
    path,info=snapshot_model('Salesforce/blip-vqa-base')
    dtype=torch.float16 if device=='cuda' else torch.float32
    model=BlipForQuestionAnswering.from_pretrained(path,torch_dtype=dtype,local_files_only=True).to(device).eval()
    processor=BlipProcessor.from_pretrained(path,local_files_only=True)
    return model,processor,device,info


def predict(model,processor,images,questions,beams,device):
    if not all(q.strip() for q in questions):raise ValueError('Empty question')
    inputs=processor(images=images,text=questions,padding=True,return_tensors='pt').to(device)
    inputs['pixel_values']=inputs.pixel_values.to(dtype=next(model.parameters()).dtype)
    if device=='cuda':torch.cuda.synchronize()
    tic=time.perf_counter()
    with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=20,num_beams=beams,do_sample=False)
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-tic
    return [s.strip() for s in processor.batch_decode(ids,skip_special_tokens=True)],seconds


def run(manifest_path,out='results/local500',batch_size=8,device='auto',settings=None):
    torch.set_num_threads(4);set_seed(42)
    manifest_path=Path(manifest_path);manifest=read_json(manifest_path)
    settings=settings or list(SETTINGS)
    model,processor,device,info=load_model(device)
    evaluator=Path(__file__).resolve().parents[1]/'third_party/vqaEval.py'
    image_hashes=read_json(manifest_path.parent/'image_hashes.json')
    hashes={x['image_id']:x for x in image_hashes}
    for i in {a['image_id'] for a in manifest}:
        if sha256(image_path(manifest_path.parent,i))!=hashes[i]['sha256']:raise ValueError('Image hash mismatch')
    out=begin_run(out,{'model':info,'n':len(manifest),'manifest_sha256':sha256(manifest_path),
        'image_hashes_sha256':sha256(manifest_path.parent/'image_hashes.json'),
        'seed':42,'device':device,'dtype':str(next(model.parameters()).dtype),
        'batch_size':batch_size,'max_new_tokens':20,'warmup_batches_per_setting':1,
        'settings':{k:SETTINGS[k] for k in settings},'evaluator_sha256':sha256(evaluator),
        'scope':'validation diagnostic with potential training overlap; no project training'})
    shutil.copy2(manifest_path,out/'eval_manifest.json')
    shutil.copy2(manifest_path.parent/'provenance.json',out/'data_provenance.json')
    shutil.copy2(manifest_path.parent/'image_hashes.json',out/'image_hashes.json')
    def images_for(batch,setting):
        result=[]
        for a in batch:
            with Image.open(image_path(manifest_path.parent,a['image_id'])) as im:
                result.append(Image.new('RGB',im.size,(127,127,127)) if setting=='blank_image' else im.convert('RGB'))
        return result
    for setting in settings:
        beams=SETTINGS[setting]['beams']
        first=manifest[:batch_size];images=images_for(first,setting)
        predict(model,processor,images,[a['question'] for a in first],beams,device)
        for im in images:im.close()
        path=out/f'predictions_{setting}.jsonl'
        existing=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        done={r['question_id']:r for r in existing}
        if len(done)!=len(existing) or not set(done).issubset({a['question_id'] for a in manifest}):
            raise ValueError('Invalid resume predictions')
        pending=[a for a in manifest if a['question_id'] not in done]
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size];tic=time.perf_counter()
            images=images_for(batch,setting)
            answers,elapsed=predict(model,processor,images,[a['question'] for a in batch],beams,device)
            e2e=time.perf_counter()-tic
            for im in images:im.close()
            with path.open('a',encoding='utf-8') as stream:
                for a,answer in zip(batch,answers):
                    row={'question_id':a['question_id'],'image_id':a['image_id'],'question':a['question'],
                         'answer':answer,'answer_type':a['answer_type'],'question_type':a['question_type'],
                         'generation_seconds_per_question':elapsed/len(batch),
                         'e2e_seconds_per_question':e2e/len(batch),'batch_n':len(batch)}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');done[a['question_id']]=row
            if len(done)%40==0 or len(done)==len(manifest):print(setting,len(done),'/',len(manifest),flush=True)
        predictions=[done[a['question_id']] for a in manifest]
        ev=evaluate(manifest,predictions)
        write_json(out/f'accuracy_{setting}.json',ev.accuracy)
        write_json(out/f'submission_{setting}.json',[{'question_id':p['question_id'],'answer':p['answer']} for p in predictions])
        print(setting,ev.accuracy['overall'],flush=True)
    from .report import build_report
    build_report(out,manifest_path.parent)
    checksums(out)
    print('Complete:',out,flush=True)
