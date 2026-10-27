"""Isolated execution/serialization utilities. No model imported here."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent
# No deployment-specific filesystem root in the portable delivery.
EXP = 'EXP-20261027-002'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)

def read_json(path):
    from binding import no_duplicate_pairs
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=no_duplicate_pairs)

def inside(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    if p != r and r not in p.parents:
        raise RuntimeError('Path outside authorized root: ' + str(p))
    return p

def server_guard(work):
    work = Path(work).expanduser().resolve()
    share = PACKAGE.parents[1]
    if work == share or work in share.parents or work == PACKAGE or PACKAGE in work.parents:
        raise ValueError('Work must be a dedicated output directory, outside variant source')
    work.mkdir(parents=True, exist_ok=True)
    for name in ['tmp', 'cache', 'logs', 'checks', 'runs']:
        inside(work / name, work).mkdir(exist_ok=True)
    for key in ['TMPDIR', 'TMP', 'TEMP']:
        os.environ[key] = str(work / 'tmp')
    tempfile.tempdir = str(work / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(work / 'cache')
    os.environ['MPLCONFIGDIR'] = str(work / 'cache' / 'matplotlib')
    os.chdir(work)
    return work

from contextlib import contextmanager
@contextmanager
def exclusive_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0: stream.write(b'0'); stream.flush()
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)

def verify_package():
    from delivery import verify
    verify()
    manifest = read_json(PACKAGE / 'manifest.json')
    for rel, expected in manifest['files'].items():
        path = inside(PACKAGE / rel, PACKAGE)
        if sha(path) != expected:
            raise RuntimeError('Package SHA mismatch: ' + rel)
    return sha(PACKAGE / 'manifest.json')

def boot_torch():
    sys.path.insert(0, str(PACKAGE / 'author'))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch

def process_record(mode):
    stat_path = Path('/proc/self/stat')
    ticks = int(stat_path.read_text().rsplit(')', 1)[1].split()[19]) if stat_path.exists() else None
    return dict(mode=mode, experiment_number=EXP, time_utc=datetime.now(timezone.utc).isoformat(),
                host=platform.node(), pid=os.getpid(), start_ticks=ticks,
                command=sys.argv, cwd=os.getcwd(), executable=sys.executable,
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                formal_training_started=False)

def capture_rng():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def save_pt(path, state):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(state, temp)
    os.replace(temp, path)

def load_pt(path, device='cpu'):
    import torch
    return torch.load(path, map_location=device, weights_only=False)

def bn_extra(model):
    return {name: {key: getattr(m, key) for key in ['eta', 'eta_test', 'test_stats_mode'] if hasattr(m, key)}
            for name, m in model.named_modules() if hasattr(m, 'test_stats_mode')}

def restore_bn_extra(model, state):
    modules = dict(model.named_modules())
    for name, values in state.items():
        for key, value in values.items():
            setattr(modules[name], key, value)

def finite_model(model):
    import torch
    bad = [n for n, v in model.state_dict().items() if not torch.isfinite(v).all()]
    if bad:
        raise FloatingPointError('Nonfinite model state: ' + str(bad))

