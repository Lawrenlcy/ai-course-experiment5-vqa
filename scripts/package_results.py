"""Package a completed result folder without models, credentials, or other runs."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.run).resolve();target=Path(args.output).resolve()
if target.is_relative_to(root):raise ValueError('ZIP output must be outside the run folder')
if not (root/'RESULTS.md').exists():raise ValueError('Run is not complete')
target.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(root.rglob('*')):
        if p.is_file():archive.write(p,p.relative_to(root))
with zipfile.ZipFile(target) as archive:
    if archive.testzip() is not None:raise ValueError('Invalid ZIP')
checksum=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
target.with_suffix('.sha256').write_text(checksum+'  '+target.name+'\n')
print(json.dumps({'file':str(target),'bytes':target.stat().st_size,'sha256':checksum}))
