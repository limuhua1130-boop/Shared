from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from common import read_json,write_json,save_pt,load_pt,capture_rng,sha,finite_model
from engine import build_model,RetainedTrainer,model_state,restore_model,geometry
from data import loaders
from runtime import config,binding,seed,inputs
from binding import require_binding,regression

def target_predict(model,cache,split,directory,bind):
    """No target truth is read by REFIT, forward, or prediction serialization."""
    model=deepcopy(model);model.eval();rows=read_json(cache/'ids.json')
    array=np.load(cache/'X.npy',mmap_mode='r');device=next(model.parameters()).device
    save_pt(directory/'before_target_adaptation.pt',dict(**model_state(model),binding=bind,rng=capture_rng()))
    selected_sha=sha(directory/'selected.pt');predictions=[]
    for domain in sorted({rows[i]['domain'] for i in split['test']}):
        ids=[i for i in split['test'] if rows[i]['domain']==domain]
        assert len(ids)==28
        x=torch.from_numpy(np.array(array[ids],copy=True)).to(device,dtype=torch.float64)
        d=torch.full((28,),domain,dtype=torch.int64)
        state=dict(**model_state(model),binding=bind,rng=capture_rng(),domain=domain,indices=ids)
        save_pt(directory/f'domain_{domain:03d}_before.pt',state)
        with torch.no_grad():
            model.domainadapt_finetune(x,None,d,None)
        finite_model(model)
        save_pt(directory/f'domain_{domain:03d}_adapted.pt',dict(**model_state(model),binding=bind,rng=capture_rng(),domain=domain,indices=ids))
        # Preserve original target DataLoader's one iterator RNG consumption.
        ds=torch.utils.data.TensorDataset(torch.arange(28))
        next(iter(torch.utils.data.DataLoader(ds,batch_size=28,num_workers=0)))
        with torch.no_grad():logits,features=model(x,d)
        pred=dict(binding=bind,selected_sha256=selected_sha,domain=domain,indices=ids,
            predictions=logits.argmax(1).cpu(),logits=logits.cpu(),geometry=geometry(model,features),truth_access=False)
        path=directory/f'domain_{domain:03d}_predictions.pt';save_pt(path,pred)
        predictions.append(dict(domain=domain,indices=ids,path=path.name,sha256=sha(path)))
        write_json(directory/'prediction_progress.json',dict(binding=bind,predictions=predictions))
        del x,pred,state
    write_json(directory/'prediction_complete.json',dict(binding=bind,predictions=predictions,complete=True,selected_sha256=selected_sha))

def run_fold(cache,work,fold,gpu,manifest,resume=False,attempt=1):
    cfg=config(fold);bind=binding(fold,cache,manifest,gpu,attempt=attempt);_,_,splits=inputs(cache)
    split=next(x for x in splits if x['fold']==fold);directory=work/'runs'/f'fold_{fold:02d}'
    from common import exclusive_lock
    with exclusive_lock(work/f'fold{fold:02d}.lock'):
        if not resume:
            directory.mkdir(parents=True,exist_ok=False);seed(fold)
            save_pt(directory/'before_model_rng.pt',dict(binding=bind,rng=capture_rng()))
            write_json(directory/'run_identity.json',bind)
        else:
            require_binding(read_json(directory/'run_identity.json'),bind)
            assert (directory/'last.pt').is_file() and not (directory/'prediction_complete.json').exists()
        model=build_model(cfg,'cuda:0');train,val=loaders(cache,split)
        if not resume:write_json(directory/'head_initialization.json',model.head_initialization)
        trainer=RetainedTrainer(cfg,'cuda:0',directory,bind)
        trainer.fit_retained(model,train,val,resume=directory/'last.pt' if resume else None)
        target_predict(model,cache,split,directory,bind)

def score_fold(cache,directory,expected_binding):
    from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score,cohen_kappa_score,confusion_matrix
    rows=read_json(cache/'ids.json');done=read_json(directory/'prediction_complete.json')
    require_binding(done['binding'],expected_binding);assert done['complete']
    binding_checks=regression(expected_binding)
    out=[];seen=[]
    for item in done['predictions']:
        p=(directory/item['path']).resolve();assert p.parent==directory.resolve() and sha(p)==item['sha256']
        pred=load_pt(p);ids=item['indices'];assert ids==pred['indices'] and all(type(i)is int for i in ids)
        require_binding(pred['binding'],expected_binding)
        assert pred['selected_sha256']==done['selected_sha256']==sha(directory/'selected.pt')
        assert pred['domain']==item['domain'] and all(rows[i]['domain']==item['domain'] for i in ids)
        y=np.array([rows[i]['label'] for i in ids]);yp=pred['predictions'].numpy()
        assert yp.shape==(28,) and yp.dtype==np.int64 and np.isfinite(pred['logits'].numpy()).all()
        assert np.all((yp>=0)&(yp<9)) and np.array_equal(yp,pred['logits'].numpy().argmax(1))
        cm=confusion_matrix(y,yp,labels=list(range(9)));assert cm.sum()==28
        out.append(dict(fold=expected_binding['fold'],domain=item['domain'],subject=rows[ids[0]]['subject'],n=28,indices=ids,
            balanced_accuracy=float(balanced_accuracy_score(y,yp)),accuracy=float(accuracy_score(y,yp)),
            macro_f1=float(f1_score(y,yp,labels=list(range(9)),average='macro',zero_division=0)),
            kappa=float(cohen_kappa_score(y,yp,labels=list(range(9)))),confusion_matrix=cm.tolist(),prediction_sha256=item['sha256']))
        seen+=ids
    assert len(seen)==len(set(seen)) and sorted(seen)==sorted(expected_binding['target_ids'])
    assert not (directory/'target_results.json').exists()
    write_json(directory/'target_results.json',out)
    write_json(directory/'completed.json',dict(binding=expected_binding,completed=True,target_domains=len(out),trials=len(seen),scored_after_prediction_freeze=True,binding_checks=binding_checks))
    return dict(domains=len(out),trials=len(seen),passed=True)
