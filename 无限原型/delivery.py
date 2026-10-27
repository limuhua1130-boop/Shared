"""Stdlib integrity, dependency inventory, and aggregation of CPU-scored results."""
from pathlib import Path
import hashlib, json, sys, statistics, math
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def verify():
    manifest=read(ROOT/'MANIFEST.json')
    for rel,digest in manifest['files'].items():
        path=(ROOT/rel).resolve()
        if ROOT not in path.parents: raise ValueError('Manifest path escapes Share')
        if sha(path)!=digest: raise ValueError('Delivery SHA mismatch: '+rel)
    return dict(passed=True,files=len(manifest['files']),manifest_sha256=sha(ROOT/'MANIFEST.json'))

def dependencies():
    from importlib.metadata import version,PackageNotFoundError
    expected={}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            k,v=line.split('==');expected[k]=v
    result={}
    for name,want in expected.items():
        try: actual=version(name)
        except PackageNotFoundError: actual=None
        result[name]=dict(expected=want,installed=actual,matched=actual is not None and actual.split('+')[0]==want)
    return dict(python=sys.version,recommended_python='3.10',packages=result,
        all_versions_match=all(x['matched'] for x in result.values()),
        note='Metadata only: no torch/CUDA import, model or numerical test was run')

def cm_metrics(cm):
    n=sum(sum(row) for row in cm);nr=[sum(row) for row in cm]
    nc=[sum(row[k] for row in cm) for k in range(9)];tp=[cm[k][k] for k in range(9)]
    acc=sum(tp)/n;pe=sum(a*b for a,b in zip(nr,nc))/(n*n)
    return dict(accuracy=acc,balanced_accuracy=statistics.mean(tp[k]/nr[k] for k in range(9) if nr[k]),
        macro_f1=statistics.mean(2*tp[k]/(nr[k]+nc[k]) if nr[k]+nc[k] else 0. for k in range(9)),
        kappa=(acc-pe)/(1-pe))

def aggregate(work,output,folds):
    work=Path(work).resolve();output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Aggregation output already exists')
    rows=[];sources=[];identities=[];predicted_ids=[]
    for fold in folds:
        d=work/'runs'/f'fold_{fold:02d}';complete=read(d/'completed.json')
        identity=read(d/'run_identity.json'); frozen=read(d/'prediction_complete.json')
        if complete['binding']!=identity or frozen['binding']!=identity: raise ValueError('Mismatched run identities')
        assert complete['completed'] and complete['scored_after_prediction_freeze'] and frozen['complete']
        assert identity['delivery']=='Share-v1' and identity['fold']==fold and identity['seed']==42+fold
        assert sha(d/'selected.pt')==frozen['selected_sha256']
        results=read(d/'target_results.json')
        assert len(results)==complete['target_domains']==len(frozen['predictions'])
        for row,item in zip(results,frozen['predictions']):
            assert row['domain']==item['domain'] and row['indices']==item['indices'] and row['n']==28
            assert row['fold']==fold and row['prediction_sha256']==item['sha256']
            p=(d/item['path']).resolve();assert p.parent==d.resolve() and sha(p)==item['sha256']
            cm=row['confusion_matrix'];assert len(cm)==9 and all(len(r)==9 for r in cm)
            assert all(type(v)is int and v>=0 for r in cm for v in r) and sum(map(sum,cm))==28
            for key,value in cm_metrics(cm).items(): assert math.isfinite(row[key]) and abs(row[key]-value)<1e-12
            rows.append(row);predicted_ids.extend(row['indices'])
        sources.append(dict(fold=fold,results_sha256=sha(d/'target_results.json'),completed_sha256=sha(d/'completed.json')))
        identities.append(identity)
    for key in ['experiment_number','manifest_sha256','config_sha256','cache_hashes','delivery']:
        assert all(b[key]==identities[0][key] for b in identities)
    assert len({r['subject'] for r in rows})==len(rows) and len(predicted_ids)==len(set(predicted_ids))
    assert sorted(predicted_ids)==sorted(i for b in identities for i in b['target_ids'])
    full=folds==list(range(10))
    if full: assert len(rows)==123 and sorted(predicted_ids)==list(range(3444))
    def stats(xs):return dict(mean=statistics.mean(xs),sample_sd=statistics.stdev(xs) if len(xs)>1 else None,n=len(xs))
    keys=['balanced_accuracy','accuracy','macro_f1','kappa']
    domain={k:stats([r[k] for r in rows]) for k in keys}
    fold={k:stats([statistics.mean(r[k] for r in rows if r['fold']==f) for f in folds]) for k in keys}
    pooled=[[sum(r['confusion_matrix'][i][j] for r in rows) for j in range(9)] for i in range(9)]
    result=dict(experiment_number=identities[0]['experiment_number'],folds=folds,full_tenfold=full,
        domains=len(rows),trials=len(predicted_ids),primary_domain_equal=domain,secondary_fold_equal=fold,
        pooled_confusion_matrix=pooled,pooled_metrics=cm_metrics(pooled),sources=sources,
        units='metrics are fractions; SD uses ddof=1; do not confuse pooled and domain/fold averages')
    write(output,result);return result
