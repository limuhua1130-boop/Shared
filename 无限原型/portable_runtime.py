"""Portable IO and run identities; no change to the frozen numerical engine."""
from pathlib import Path
import os, sys, random, subprocess
from common import PACKAGE, sha, read_json, write_json, inside, server_guard, verify_package, boot_torch

EXP=read_json(PACKAGE/'config.json')['experiment_number']
UUID={}
_INPUTS={}

def bootstrap(work,gpu=None):
    if not __debug__: raise RuntimeError('Do not use python -O: scientific guards require assertions')
    manifest=verify_package()
    if gpu is None:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        if gpu not in (0,3): raise ValueError('Physical GPU must be 0 or 3')
        actual=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if not actual.startswith('GPU-') or '\n' in actual: raise RuntimeError('Expected exactly one GPU UUID')
        UUID[gpu]=actual
        os.environ['CUDA_VISIBLE_DEVICES']=actual
    work=server_guard(Path(work))
    torch=boot_torch()
    if gpu is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Expected one visible CUDA device after physical UUID binding')
    from trainer_metric_compat import install
    install()
    write_json(work/f'environment_pid{os.getpid()}.json',dict(python=sys.version,torch=torch.__version__,physical_gpu=gpu,
        gpu_uuid=UUID.get(gpu),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],manifest_sha256=manifest))
    return torch,work,manifest

def inputs(cache,full_hash=False):
    cache=Path(cache).expanduser().resolve()
    if cache in _INPUTS and not full_hash: return _INPUTS[cache]
    import numpy as np
    prepared=read_json(cache/'prepared.json')
    hashes=prepared['cache_hashes']
    if set(hashes)!= {'X.npy','ids.json','splits.json'}: raise ValueError('Unexpected prepared cache schema')
    for name,digest in hashes.items():
        if sha(inside(cache/name,cache))!=digest: raise ValueError('Cache SHA mismatch: '+name)
    array=np.load(cache/'X.npy',mmap_mode='r')
    if array.shape!=(3444,32,7500) or array.dtype!=np.float64: raise ValueError('Expected float64 X[3444,32,7500]')
    rows=read_json(cache/'ids.json'); splits=read_json(cache/'splits.json')
    assert len(rows)==3444 and [s['fold'] for s in splits]==list(range(10))
    assert [r['index'] for r in rows]==list(range(3444))
    assert len({r['sample_id'] for r in rows})==3444
    labels=[k for k,n in enumerate([3,3,3,3,4,3,3,3,3]) for _ in range(n)]
    expected_subjects=sorted(range(123),key=lambda i:'subject_'+str(i))
    domain_names=sorted(str(i)+'_1' for i in range(123))
    domain_map={name:i for i,name in enumerate(domain_names)}
    for i,r in enumerate(rows):
        subject=expected_subjects[i//28]; video=i%28
        assert (r['subject'],r['session'],r['video'],r['label'],r['domain'])==(subject,1,video,labels[video],domain_map[str(subject)+'_1'])
        assert r['window']=='whole_trial' and r['start']==0 and r['end']==7500
    all_test=[]
    for s in splits:
        tr,va,te=[s[k] for k in ('train','val','test')]
        assert all(type(i)is int and 0<=i<3444 for ids in (tr,va,te) for i in ids)
        assert len(tr)+len(va)+len(te)==3444 and len(set(tr+va+te))==3444
        assert not {rows[i]['subject'] for i in te}&{rows[i]['subject'] for i in tr+va}
        assert len(va) in (616,622)
        all_test+=te
    assert sorted(all_test)==list(range(3444))
    # Reconstruct tiny ID splits, not EA/data. Prevent a different split being called this protocol.
    from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
    y=np.array([r['label'] for r in rows]); subjects=np.array([r['subject'] for r in rows])
    dummy=np.zeros((3444,1)); rng=np.random.RandomState(42)
    for f,(fit,test) in enumerate(GroupKFold(10).split(dummy,y,subjects)):
        tr,va=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=rng).split(dummy[fit],y[fit]))
        assert splits[f]['train']==fit[tr].tolist() and splits[f]['val']==fit[va].tolist() and splits[f]['test']==test.tolist()
    info={'cache':{k:{'sha256':v} for k,v in hashes.items()}}
    info['cache']['prepared.json']={'sha256':sha(cache/'prepared.json')}
    _INPUTS[cache]=(info,rows,splits)
    return _INPUTS[cache]

def seed(fold):
    import numpy as np, torch
    random.seed(42+fold);np.random.seed(42+fold);torch.manual_seed(42+fold)
    if torch.cuda.is_available():
        assert torch.cuda.device_count()==1;torch.cuda.manual_seed_all(42+fold)

def attempt_label(attempt):
    if type(attempt)is not int or attempt<1: raise ValueError('Attempt must be a positive integer')
    return f'R{attempt:02d}'

def claim_attempt(stored):
    label=stored['attempt']
    if type(label)is not str or len(label)<3 or label[0]!='R' or not label[1:].isascii() or not label[1:].isdecimal():
        raise ValueError('Stored attempt is not a canonical RNN identity')
    value=int(label[1:])
    if attempt_label(value)!=label: raise ValueError('Stored attempt is not canonical')
    return value

def binding(fold,cache,manifest,gpu,attempt=1):
    label=attempt_label(attempt);info,rows,splits=inputs(cache);split=splits[fold]
    if gpu is not None and gpu not in UUID: raise ValueError('Missing physical GPU UUID; CPU scoring must use saved identity')
    return dict(experiment_number=EXP,task='TASK-20261027-007',delivery='Share-v1',fold=fold,seed=42+fold,attempt=label,
        run_number=f'{EXP}-F{fold:02d}-S{42+fold}-{label}',manifest_sha256=manifest,
        config_sha256=sha(PACKAGE/'config.json'),source_manifest_sha256=sha(PACKAGE/'source_manifest.json'),
        inputs_sha256=sha(Path(cache)/'prepared.json'),cache_hashes={k:v['sha256'] for k,v in info['cache'].items()},
        train_ids=split['train'],val_ids=split['val'],target_ids=split['test'],
        target_order=sorted({rows[i]['domain'] for i in split['test']}),
        target_ids_by_domain={str(d):[i for i in split['test'] if rows[i]['domain']==d] for d in sorted({rows[i]['domain'] for i in split['test']})},
        physical_gpu=gpu,gpu_uuid=UUID.get(gpu),
        geometry='Lorentz curvature -1' if config(fold)['variant']=='D1' else 'equivalent unit Poincare curvature -1',
        initialization='fresh, no trained weights')

def config(fold):
    cfg=read_json(PACKAGE/'config.json')
    cfg.update(fold=fold,actual_fold_seed=42+fold,torch_seed=42+fold,python_seed=42+fold,numpy_training_seed=42+fold)
    return cfg
