"""Single process per variant/stage; prevents collisions of the two authors' lib packages."""
from pathlib import Path
import sys, json, os
ROOT=Path(__file__).resolve().parent

def execute(a):
    variant=ROOT/'variants'/a.variant
    sys.path.insert(0,str(variant))
    cache=Path(a.cache).expanduser().resolve();work=Path(a.work).expanduser().resolve()
    checkpoint=Path(a.checkpoint).expanduser().resolve() if a.checkpoint else None
    output=Path(a.output).expanduser().resolve() if a.output else None
    indices=Path(a.indices_json).expanduser().resolve() if a.indices_json else None
    from runtime import bootstrap,inputs,binding,config,claim_attempt,UUID
    from common import read_json,write_json,process_record,inside
    torch,work,manifest=bootstrap(work,a.gpu)
    from execution_compat import install
    receipt_work=work/'compatibility'/f'pid_{os.getpid()}'
    receipt_work.mkdir(parents=True,exist_ok=False)
    receipt=install(torch,receipt_work,manifest,a.mode)
    try:
        info,rows,splits=inputs(cache)
        write_json(work/('process_'+a.mode+f'_fold{a.fold}.json'),process_record(a.mode))
        if a.mode=='preflight':
            from binding import regression
            write_json(work/'checks/preflight.json',dict(passed=True,manifest=manifest,
                full_cache_hash=True,validation_sizes=[len(s['val']) for s in splits],
                binding_checks=regression(binding(0,cache,manifest,None))))
        elif a.mode in ('algebra','smoke','capacity'):
            from validate import validate
            validate(cache,work,a.mode,a.gpu)
        elif a.mode=='train':
            from workflow import run_fold
            run_fold(cache,work,a.fold,a.gpu,manifest,a.resume,attempt=a.attempt)
        elif a.mode=='score':
            from workflow import score_fold
            from binding import require_binding
            directory=work/'runs'/f'fold_{a.fold:02d}';stored=read_json(directory/'run_identity.json')
            UUID[stored['physical_gpu']]=stored['gpu_uuid']
            expected=binding(a.fold,cache,manifest,stored['physical_gpu'],attempt=claim_attempt(stored))
            require_binding(stored,expected)
            print(json.dumps(score_fold(cache,directory,expected)),flush=True)
        elif a.mode=='evaluation-score':
            from evaluation_check import score_evaluation
            stored=read_json(work/'checks'/a.check_mode/'evaluation/run_identity.json')
            UUID[stored['physical_gpu']]=stored['gpu_uuid']
            score_evaluation(cache,work,a.check_mode,manifest)
        else:
            from engine import export_geometry
            export_geometry(config(a.fold),cache,inside(checkpoint,work),inside(output,work),
                'cuda:0' if a.gpu is not None else 'cpu',read_json(indices) if indices else None)
    finally: receipt()
