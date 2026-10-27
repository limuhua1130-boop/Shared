"""Print every effective hyperparameter, then invoke the frozen complete pipeline."""
from pathlib import Path
import argparse,sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1]

def main(variant):
    p=argparse.ArgumentParser(description=variant+' original max100 / lr.001; train -> REFIT -> frozen predictions -> separate CPU score')
    p.add_argument('--cache',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--folds',nargs='+',type=int,choices=range(10),default=list(range(10)))
    p.add_argument('--gpus',nargs='+',type=int,choices=[0,3],default=[0,3])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    print((ROOT/'hyperparameters'/f'{variant}.json').read_text(encoding='utf-8'),flush=True)
    command=[sys.executable,str(ROOT/'share.py'),'schedule','--variants',variant,'--cache',a.cache,
        '--output-root',a.output_root,'--folds',*[str(f) for f in a.folds],'--gpus',*[str(g) for g in a.gpus]]
    if a.dry_run:command.append('--dry-run')
    raise SystemExit(subprocess.call(command))
