import concurrent.futures
import json
import random
import zipfile
from pathlib import Path
from PIL import Image
from .common import fetch, read_json, write_json, sha256

SOURCES={
    'questions':('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip','v2_OpenEnded_mscoco_val2014_questions.json'),
    'annotations':('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip','v2_mscoco_val2014_annotations.json'),
}


def image_path(root,image_id):
    return Path(root)/'images'/f'COCO_val2014_{image_id:012d}.jpg'


def prepare(root='data', n=500, seed=42):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/f'manifest_n{n}_seed{seed}.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
    else:
        objects={};sources={}
        for kind,(url,member) in SOURCES.items():
            path=fetch(url,root/(kind+'.zip'))
            with zipfile.ZipFile(path) as z:objects[kind]=json.loads(z.read(member))
            sources[kind]={'url':url,'sha256':sha256(path)}
        questions={q['question_id']:q['question'] for q in objects['questions']['questions']}
        selected=random.Random(seed).sample(sorted(objects['annotations']['annotations'],key=lambda a:a['question_id']),n)
        manifest=[dict(a,question=questions[a['question_id']]) for a in selected]
        write_json(manifest_path,manifest)
        write_json(root/'provenance.json',{'sources':sources,'n':n,'seed':seed,
            'population':len(objects['annotations']['annotations']),
            'selection':'random.sample(sorted(annotations, question_id), n)',
            'scope':'VQAv2 val2014 diagnostic; potential BLIP training overlap'})
    def download(i):
        path=image_path(root,i)
        fetch(f'https://s3.amazonaws.com/images.cocodataset.org/val2014/{path.name}',path)
        with Image.open(path) as image:image.verify()
        return {'image_id':i,'file':path.relative_to(root).as_posix(),'sha256':sha256(path)}
    ids=sorted({a['image_id'] for a in manifest});failures=[];hashes=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(download,i):i for i in ids}
        for k,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:hashes.append(f.result())
            except Exception as e:failures.append({'image_id':futures[f],'error':str(e)})
            if k%50==0:print('Images',k,'/',len(ids),flush=True)
    write_json(root/'download_failures.json',failures)
    if failures:raise RuntimeError(f'{len(failures)} images failed; rerun prepare, never replace samples')
    write_json(root/'image_hashes.json',sorted(hashes,key=lambda x:x['image_id']))
    return manifest_path
