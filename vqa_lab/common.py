"""Portable provenance, downloading, immutable run identity, and plotting helpers."""
import hashlib
import json
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path, attempts=4):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    for attempt in range(attempts):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                append = response.status == 206 and offset > 0
                expected = int(response.headers.get('Content-Length', 0))
                received = 0
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned the wrong byte range')
                with temp.open('ab' if append else 'wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                if expected and received != expected:
                    raise IOError(f'Incomplete download: {received}/{expected} bytes; will resume')
            temp.replace(path)
            return path
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def environment():
    import importlib.metadata
    import torch
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'os': platform.platform(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'packages': {p: importlib.metadata.version(p) for p in
                     ['transformers', 'numpy', 'scipy', 'librosa', 'pandas', 'Pillow']},
    }


def begin_run(out, config):
    """Reject mixed checkpoints, datasets, settings, or code on resume."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity = dict(config, code_sha256=code)
    path = out / 'run_config.json'
    if path.exists():
        if read_json(path) != identity:
            raise ValueError('Run configuration/source changed. Use a new --out directory.')
        recorded=read_json(out/'environment.json')
        current=environment()
        if any(recorded[k] != current[k] for k in ['python','torch','cuda','gpu','packages']):
            raise ValueError('Runtime environment changed. Use a new --out directory.')
    else:
        write_json(path, identity)
        write_json(out / 'environment.json', environment())
        import shutil
        snapshot=out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob('*.py'):
            shutil.copy2(source,snapshot/source.name)
    return out


def snapshot_model(model_id, root='models'):
    """Resolve once, then lock every model file to an immutable Hub commit."""
    lock = Path(root) / (model_id.replace('/', '--') + '.json')
    shared_lock = Path('configs/models.json')
    pinned = read_json(shared_lock) if shared_lock.exists() else {}
    if model_id in pinned:
        info = {'model_id':model_id, 'revision':pinned[model_id]}
        if lock.exists() and read_json(lock) != info:
            raise ValueError('Local model lock differs from configs/models.json')
        write_json(lock, info)
    elif lock.exists():
        info = read_json(lock)
    else:
        with urllib.request.urlopen('https://huggingface.co/api/models/' + model_id, timeout=60) as r:
            metadata = json.load(r)
        info = {'model_id': model_id, 'revision': metadata['sha']}
        write_json(lock, info)
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, revision=info['revision'],
                             allow_patterns=['*.json', '*.txt', '*.safetensors'],
                             local_dir=Path(root) / model_id.replace('/', '--'))
    return path, info


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white'})
    return plt


def save_figure(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (name + '.png'), dpi=220, bbox_inches='tight')
    fig.savefig(folder / (name + '.svg'), bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def checksums(out):
    out = Path(out)
    write_json(out / 'checksums_sha256.json', {
        p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name != 'checksums_sha256.json'
    })
