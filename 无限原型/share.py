#!/usr/bin/env python
"""D1/D2 frozen100 delivery. All --help/dry-run/metadata actions use only stdlib."""
from pathlib import Path
import argparse, sys, os, json, subprocess
ROOT=Path(__file__).resolve().parent

def parser():
    p=argparse.ArgumentParser(description='D1 Lorentz / D2 equivalent Poincare, original max100 and lr=.001')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Verify all delivery SHA256 values, no scientific imports')
    sub.add_parser('dependencies',help='Read installed package metadata only')
    ps=sub.add_parser('params',help='Print frozen complete effective hyperparameters')
    ps.add_argument('--variant',choices=['D1','D2'],required=True)
    prep=sub.add_parser('prepare',help='Raw sub000.pkl...sub122.pkl -> whole-trial EA/cache')
    prep.add_argument('--raw',required=True);prep.add_argument('--cache',required=True)
    run=sub.add_parser('run',help='Execute one explicitly chosen stage')
    run.add_argument('--variant',choices=['D1','D2'],required=True)
    run.add_argument('--mode',choices=['preflight','algebra','smoke','capacity','train','score','evaluation-score','export'],required=True)
    run.add_argument('--cache',required=True);run.add_argument('--work',required=True)
    run.add_argument('--gpu',type=int,choices=[0,3]);run.add_argument('--fold',type=int,choices=range(10))
    run.add_argument('--attempt',type=int,default=1);run.add_argument('--resume',action='store_true')
    run.add_argument('--check-mode',choices=['smoke','capacity'])
    run.add_argument('--checkpoint');run.add_argument('--output');run.add_argument('--indices-json')
    schedule=sub.add_parser('schedule',help='Run each chosen fold: training+REFIT+frozen prediction, then separate CPU score')
    schedule.add_argument('--variants',nargs='+',choices=['D1','D2'],default=['D1','D2'])
    schedule.add_argument('--folds',nargs='+',type=int,choices=range(10),default=list(range(10)))
    schedule.add_argument('--gpus',nargs='+',type=int,choices=[0,3],default=[0,3])
    schedule.add_argument('--cache',required=True);schedule.add_argument('--output-root',required=True)
    schedule.add_argument('--dry-run',action='store_true',help='Print commands; no directories/GPU queries/imports/processes')
    agg=sub.add_parser('aggregate',help='Verify and aggregate separately CPU-scored folds')
    agg.add_argument('--work',required=True);agg.add_argument('--output',required=True)
    agg.add_argument('--folds',nargs='+',type=int,choices=range(10),default=list(range(10)))
    return p

def schedule(a):
    if len(set(a.gpus))!=len(a.gpus) or len(set(a.variants))!=len(a.variants) or len(set(a.folds))!=len(a.folds):
        raise ValueError('No duplicate GPU, variant or fold')
    cache=Path(a.cache).expanduser().resolve();out=Path(a.output_root).expanduser().resolve()
    jobs=[(v,f) for f in a.folds for v in a.variants]
    queues={g:[] for g in a.gpus}
    for i,(variant,fold) in enumerate(jobs):
        gpu=a.gpus[i%len(a.gpus)];work=out/variant
        base=[sys.executable,str(ROOT/'share.py'),'run','--variant',variant,'--cache',str(cache),'--work',str(work),'--fold',str(fold)]
        train=base+['--mode','train','--gpu',str(gpu)]
        score=base+['--mode','score']
        queues[gpu].append(dict(variant=variant,fold=fold,train=train,score=score))
    if a.dry_run:
        print(json.dumps(dict(one_complete_pipeline_per_gpu=True,jobs=queues),ensure_ascii=False,indent=2));return
    from delivery import verify,write
    verify()
    for variant,fold in jobs:
        if (out/variant/'runs'/f'fold_{fold:02d}').exists():
            raise FileExistsError('Fresh schedule refuses an existing fold; use explicit run --resume or a new output root')
    # Locks span train and score; separate invocations sharing this output root cannot oversubscribe.
    sys.path.insert(0,str(ROOT/'variants/D1'))
    from common import exclusive_lock
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    def worker(gpu,jobs):
        for job in jobs:
            logdir=out/job['variant']/'logs';logdir.mkdir(parents=True,exist_ok=True)
            for mode in ('train','score'):
                log=logdir/f"fold_{job['fold']:02d}_{mode}.log"
                with log.open('x',encoding='utf-8') as f:
                    p=subprocess.run(job[mode],stdout=f,stderr=subprocess.STDOUT,env=env)
                write(log.with_suffix('.exit.json'),dict(returncode=p.returncode,command=job[mode],physical_gpu=gpu))
                if p.returncode: raise RuntimeError('Stage failed; no later fold on this GPU: '+str(log))
    with ExitStack() as locks:
        for gpu in a.gpus: locks.enter_context(exclusive_lock(out/f'physical_gpu{gpu}.lock'))
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            futures=[pool.submit(worker,gpu,queue) for gpu,queue in queues.items()]
            for f in futures: f.result()

def main():
    p=parser();a=p.parse_args()
    if not __debug__: p.error('Do not use python -O')
    sys.dont_write_bytecode=True
    if a.command=='run':
        if a.attempt<1:p.error('--attempt must be positive')
        if a.mode in ('train','score','export') and a.fold is None:p.error('This mode needs --fold')
        if a.mode in ('train','capacity') and a.gpu is None:p.error('This mode needs --gpu 0 or 3')
        if a.mode in ('preflight','algebra','score','evaluation-score') and a.gpu is not None:p.error('This mode is CPU-only')
        if a.resume and a.mode!='train':p.error('--resume is for train only')
        if a.mode=='evaluation-score' and a.check_mode is None:p.error('Specify --check-mode')
        if a.mode=='export' and (not a.checkpoint or not a.output):p.error('Export needs --checkpoint and --output')
        from stage import execute
        execute(a)
    elif a.command=='schedule':schedule(a)
    elif a.command=='prepare':
        from delivery import verify,read
        verify();os.environ['CUDA_VISIBLE_DEVICES']=''
        sys.path.insert(0,str(ROOT/'variants/D1'))
        from common import boot_torch,exclusive_lock
        boot_torch()
        from data_builder import prepare_data
        cache=Path(a.cache).expanduser().resolve()
        with exclusive_lock(cache.parent/(cache.name+'.prepare.lock')):
            prepare_data(Path(a.raw).expanduser().resolve(),cache,read(ROOT/'data_protocol.json'))
    else:
        from delivery import verify,dependencies,read,aggregate
        if a.command=='verify':result=verify()
        elif a.command=='dependencies':result=dependencies()
        elif a.command=='params':result=read(ROOT/'hyperparameters'/f'{a.variant}.json')
        else:
            if len(set(a.folds))!=len(a.folds):p.error('Duplicate fold')
            result=aggregate(a.work,a.output,sorted(a.folds))
        print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
