"""Actual target prediction serialization, then separately invoked CPU scoring."""
from copy import deepcopy
from common import read_json,write_json,save_pt,sha,PACKAGE,capture_rng
from runtime import binding,claim_attempt
from engine import model_state
from workflow import target_predict,score_fold
from binding import require_binding

def expected(cache,manifest,gpu,domain):
    b=binding(0,cache,manifest,gpu)
    ids=b['target_ids_by_domain'][str(domain)]
    b.update(target_ids=ids,target_order=[domain],target_ids_by_domain={str(domain):ids},
        validation_only=True,not_formal_result=True)
    return b

def prepare_evaluation(model,cache,split,directory,device):
    # Model already REFIT by lifecycle check; restored before-state is exercised
    # there. This distinct disposable check tests actual production IO contract.
    from common import load_pt
    from engine import restore_model
    restore_model(model,load_pt(directory/'before_refit.pt',device))
    import os
    from runtime import UUID
    gpu=next((g for g,u in UUID.items() if os.environ.get('CUDA_VISIBLE_DEVICES')==u),None)
    rows=read_json(cache/'ids.json');domain=rows[split['test'][0]]['domain']
    b=expected(cache,sha(PACKAGE/'manifest.json'),gpu,domain)
    dest=directory/'evaluation';dest.mkdir(exist_ok=False)
    save_pt(dest/'selected.pt',dict(**model_state(model),binding=b,rng=capture_rng()))
    write_json(dest/'run_identity.json',b)
    small=deepcopy(split);small['test']=b['target_ids']
    target_predict(model,cache,small,dest,b)
    write_json(dest/'awaiting_CPU_score.json',dict(passed_prediction=True,formal_result=False,
        source_mode=directory.name,gpu=gpu,domain=domain))

def score_evaluation(cache,work,mode,manifest):
    directory=work/'checks'/mode/'evaluation'
    saved=read_json(directory/'run_identity.json')
    b=expected(cache,manifest,saved['physical_gpu'],saved['target_order'][0])
    require_binding(saved,b)
    result=score_fold(cache,directory,b)
    # Independent confusion-matrix oracle, no sklearn metric calls here.
    import numpy as np
    from common import load_pt
    rows=read_json(cache/'ids.json');item=read_json(directory/'prediction_complete.json')['predictions'][0]
    frozen=load_pt(directory/item['path']);cm=np.zeros((9,9),dtype=np.int64)
    for i,p in zip(frozen['indices'],frozen['predictions'].tolist()):cm[rows[i]['label'],p]+=1
    tp=cm.diagonal();nr=cm.sum(1);nc=cm.sum(0);n=cm.sum()
    acc=float(tp.sum()/n);ba=float((tp[nr>0]/nr[nr>0]).mean())
    f1=float(np.divide(2*tp,nr+nc,out=np.zeros(9),where=(nr+nc)>0).mean())
    pe=float((nr*nc).sum()/n**2);k=(acc-pe)/(1-pe)
    actual=read_json(directory/'target_results.json')[0]
    for name,value in [('accuracy',acc),('balanced_accuracy',ba),('macro_f1',f1),('kappa',k)]:
        assert abs(actual[name]-value)<1e-12,(name,actual[name],value)
    assert actual['confusion_matrix']==cm.tolist()
    write_json(directory/'CPU_score_regression.json',dict(passed=True,result=result,
        independently_reconstructed_metrics=True,formal_result=False))
