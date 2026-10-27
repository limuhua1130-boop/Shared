"""03-run affected-scope checks. Fails closed; JSON is written in finally."""
from pathlib import Path
from copy import deepcopy
import json,time,math
import torch
import numpy as np
from common import write_json,save_pt,load_pt,capture_rng,restore_rng,finite_model
from runtime import seed,config,inputs,binding
from engine import build_model,model_state,restore_model,geometry,RetainedTrainer
from optimizer_checkpoint import restore_optimizer,exact_state_equal
from objective import swd_loss
from data import loaders

def exact_state_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if type(a)is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact_state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact_state_equal(x,y) for x,y in zip(a,b))
    return a==b

# CUDA replay computes new floating-point results; loading stored state remains exact.
CUDA_REPLAY_ATOL=1e-14
CUDA_REPLAY_RTOL=1e-12

def continuation_compare(a,b,cuda=False):
    """Finite float64 tensors may round; all structure/non-tensor metadata stays exact."""
    atol=CUDA_REPLAY_ATOL if cuda else 0.
    rtol=CUDA_REPLAY_RTOL if cuda else 0.
    report=dict(passed=True,bitwise_equal=bool(exact_state_equal(a,b)),atol=atol,rtol=rtol,
        rule='abs(a-b)<=atol+rtol*max(abs(a),abs(b))',
        relative_error='abs(a-b)/max(abs(a),abs(b),float64.tiny)',
        max_abs=0.,max_rel=0.,floating_tensor_leaves=0,differences=[],failures=[])
    def fail(path,reason):
        report['passed']=False;report['failures'].append(dict(path=path,reason=reason))
    def walk(x,y,path):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or x.dtype!=y.dtype or x.shape!=y.shape:
                fail(path,'tensor type/shape/dtype mismatch');return
            x=x.detach().cpu();y=y.detach().cpu()
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                fail(path,'nonfinite tensor');return
            if not x.is_floating_point():
                if not torch.equal(x,y):fail(path,'non-floating tensor must be exact')
                return
            if x.dtype!=torch.float64:
                fail(path,'replay floating tensor must remain float64');return
            report['floating_tensor_leaves']+=1
            if not x.numel():return
            delta=(x-y).abs();scale=torch.maximum(x.abs(),y.abs())
            if not bool(torch.isfinite(delta).all()):
                fail(path,'nonfinite difference');return
            max_abs=float(delta.max());max_rel=float((delta/scale.clamp_min(torch.finfo(x.dtype).tiny)).max())
            report['max_abs']=max(report['max_abs'],max_abs);report['max_rel']=max(report['max_rel'],max_rel)
            within=bool((delta<=atol+rtol*scale).all())
            if max_abs:
                report['differences'].append(dict(path=path,shape=list(x.shape),dtype=str(x.dtype),
                    unequal_count=int((x!=y).sum()),elements=x.numel(),max_abs=max_abs,max_rel=max_rel,within_tolerance=within))
            if not within:fail(path,'floating difference exceeds fixed tolerance')
        elif type(x)is not type(y):fail(path,'metadata type mismatch')
        elif isinstance(x,dict):
            if x.keys()!=y.keys():fail(path,'dictionary keys mismatch');return
            for k in x:walk(x[k],y[k],path+'['+repr(k)+']')
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):fail(path,'sequence length mismatch');return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'['+str(i)+']')
        else:
            if isinstance(x,(float,np.floating)) and not bool(np.isfinite(x) and np.isfinite(y)):
                fail(path,'nonfinite scalar metadata')
            elif not exact_state_equal(x,y):fail(path,'metadata must be exact')
    walk(a,b,'$');return report


def check_close(name,a,b,results,atol=1e-7,rtol=1e-6):
    assert a.shape==b.shape,(name,a.shape,b.shape)
    err=float((a-b).detach().abs().max());ok=torch.allclose(a,b,atol=atol,rtol=rtol)
    results.append(dict(check=name,max_abs=err,atol=atol,rtol=rtol,passed=ok))
    if not ok:raise AssertionError(results[-1])

def exercise(cache,directory,device,results,capacity=False):
    seed(0);cfg=config(0);model=build_model(cfg,device);_,rows,splits=inputs(cache);split=splits[0]
    train,_=loaders(cache,split);f,y=next(iter(train));ids=train.batch_ids.copy()
    if not capacity:f={k:v[:4] for k,v in f.items()};y=y[:4];ids=ids[:4]
    f['inputs']=f['inputs'].to(device);y=y.to(device)
    trainer=RetainedTrainer(cfg,device,directory/'discardable',{'fold':0,'validation':True})
    opt=model.configure_optimizers(lr=.001,weight_decay=.001);trainer.optimizer=opt
    for cb in trainer.callbacks:cb.on_fit_start(trainer,model);cb.on_train_epoch_start(trainer,model)
    peaks=[]
    def phase(name,fn):
        if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        t=time.monotonic();v=fn()
        if device.startswith('cuda'):
            torch.cuda.synchronize();peaks.append(dict(stage=name,allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-t))
        return v
    # Only the validator saves/compares gradients; the scientific update is unchanged.
    def update(capture=False):
        model.train();out,feat=model(**f);loss=torch.nn.functional.cross_entropy(out,y)+.01*swd_loss(feat,f['domains'],model.manifold)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        observed=None
        if capture:
            observed=dict(logits=out.detach().cpu().clone(),features=feat.detach().cpu().clone(),
                gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()})
        opt.step();opt.zero_grad();finite_model(model);return float(loss.detach()),geometry(model,feat),observed
    initial=model_state(model);loss,g,_=phase('B50_train_update' if capacity else 'B4_train_update',update)
    changed=sum(not torch.equal(v,model.state_dict()[n]) for n,v in initial['model'].items())
    assert changed>0
    trainer.current_step=1;state=trainer.snapshot(model,0,False)
    state['validation_batch_ids']=ids.copy();state['validation_domains']=f['domains'].tolist()
    save_pt(directory/'one_update.pt',state);saved=load_pt(directory/'one_update.pt',device)
    def restore_exact(branch):
        restore_model(model,saved);restore_optimizer(opt,saved['optimizer']);restore_rng(saved['rng'])
        current=model_state(model)
        checks=dict(model_BN_alias_and_mode_exact=exact_state_equal(current,{k:saved[k] for k in current}),
            optimizer_exact=exact_state_equal(opt.state_dict(),saved['optimizer']),
            RNG_exact=exact_state_equal(capture_rng(),saved['rng']),
            no_pending_gradients=all(p.grad is None for p in model.parameters()),
            batch_IDs_exact=ids==saved['validation_batch_ids'] and len(ids)==len(y)
                and all(i in split['train'] for i in ids),
            domains_exact=f['domains'].tolist()==saved['validation_domains'])
        results.append(dict(check='strict_load_before_continuation_'+branch,passed=all(checks.values()),**checks))
        assert all(checks.values()),checks
    restore_exact('A')
    l1,_,observed1=phase('same_input_continuation_A',lambda:update(capture=True))
    s1=model_state(model);o1=deepcopy(opt.state_dict());r1=capture_rng()
    restore_exact('B')
    l2,_,observed2=phase('same_input_continuation_B',lambda:update(capture=True))
    s2=model_state(model);o2=opt.state_dict();r2=capture_rng();cuda=device.startswith('cuda')
    original=dict(loss=l1==l2,model=exact_state_equal(s1,s2),optimizer=exact_state_equal(o1,o2),RNG=exact_state_equal(r1,r2))
    comparisons=dict(loss=continuation_compare(torch.tensor(l1,dtype=torch.float64),torch.tensor(l2,dtype=torch.float64),cuda),
        model=continuation_compare(s1,s2,cuda),optimizer=continuation_compare(o1,o2,cuda),
        forward_and_gradients=continuation_compare(observed1,observed2,cuda))
    accepted=original['RNG'] and all(v['passed'] for v in comparisons.values())
    results.append(dict(check='numerical_CUDA_or_exact_CPU_continuation',passed=accepted,
        original_bitwise_checks=original,original_bitwise_passed=all(original.values()),
        RNG_required_exact=True,comparisons=comparisons,loss_A=l1,loss_B=l2,changed_states=changed,batch_ids=ids))
    assert accepted,comparisons
    # Release added validator copies before full source validation/target REFIT.
    del observed1,observed2,s2,o2,r2,comparisons

    # Exercise ES save/restore, same author trigger including min-epoch boundary.
    trainer.records=[dict(epoch=0,val_loss=1.)];trainer.es.on_train_epoch_end(trainer,model)
    saved_es=model_state(model);trainer.es.on_fit_end(trainer,model)
    assert exact_state_equal(saved_es,model_state(model))
    trainer.current_epoch=70;trainer.stop_fit();assert not trainer.stop_fit_
    trainer.current_epoch=71;trainer.stop_fit();assert trainer.stop_fit_
    results.append(dict(check='ES_selected_restore_and_min_epoch_rule',passed=True))
    del f,y,initial,saved,s1,o1,r1,state
    array=np.load(cache/'X.npy',mmap_mode='r')
    maxsplit=max(splits,key=lambda s:len(s['val']));vids=maxsplit['val'] if capacity else maxsplit['val'][:4]
    vx=torch.from_numpy(np.array(array[vids],copy=True)).to(device);vd=torch.tensor([rows[i]['domain'] for i in vids]);model.eval()
    def validation():
        with torch.no_grad():out,feat=model(vx,vd)
        assert out.shape==(len(vids),9);return geometry(model,feat)
    phase('whole_source_validation_'+str(len(vids)),validation)
    vy=torch.tensor([rows[i]['label'] for i in vids])
    def actual_test():
        result=trainer.test(model,[({'inputs':vx,'domains':vd},vy)])
        assert all(np.isfinite(v) for v in result.values())
        return result
    eval_result=phase('author_Trainer_test_and_metric_bridge',actual_test)
    results.append(dict(check='author_Trainer_test_val_CE_metric',passed=True,result=eval_result))
    del vx,vd,vy
    domain=rows[split['test'][0]]['domain'];tids=[i for i in split['test'] if rows[i]['domain']==domain]
    tx=torch.from_numpy(np.array(array[tids],copy=True)).to(device);td=torch.full((28,),domain,dtype=torch.int64)
    pre=model_state(model);save_pt(directory/'before_refit.pt',dict(**pre,indices=tids))
    class ForbiddenTruth:
        def __getattribute__(self,name):raise AssertionError('Target truth inspected')
    def refit():
        with torch.no_grad():model.domainadapt_finetune(tx,ForbiddenTruth(),td,None)
    phase('target28_unlabelled_REFIT',refit);post=model_state(model)
    save_pt(directory/'after_refit.pt',dict(**post,indices=tids))
    restore_model(model,load_pt(directory/'before_refit.pt',device))
    with torch.no_grad():model.domainadapt_finetune(tx,None,td,None)
    assert exact_state_equal(post,model_state(model))
    assert all(torch.equal(pre['model'][n],post['model'][n]) for n,_ in model.named_parameters())
    def predict():
        with torch.no_grad():out,feat=model(tx,td)
        assert torch.isfinite(out).all();return geometry(model,feat)
    geom=phase('target28_prediction',predict)
    restore_model(model,load_pt(directory/'after_refit.pt',device));assert exact_state_equal(post,model_state(model))
    results.append(dict(check='target_truth_isolation_REFIT_restore',passed=True,target_count=28,domain=domain,validation_count=len(vids)))
    write_json(directory/'capacity.json',dict(peaks=peaks,peak_reserved_bytes=max([x['reserved_bytes'] for x in peaks],default=0),
        below23GiB=max([x['reserved_bytes'] for x in peaks],default=2**63)<23*2**30 if capacity else None,
        process_peak_requires03_external_sampling=True,dual_instance_capacity_not_claimed=True))
    save_pt(directory/'native_geometry.pt',dict(train=g,target=geom,train_ids=ids,target_ids=tids))
    from evaluation_check import prepare_evaluation
    prepare_evaluation(model,cache,split,directory,device)

def validate(cache,work,mode,gpu=None):
    dest=work/'checks'/mode;dest.mkdir(parents=True,exist_ok=False);results=[];passed=False
    try:
        if mode=='algebra':
            from head_checks import algebra,paired_backbone
            algebra(results);paired_backbone(cache,results)
        else:exercise(cache,dest,'cuda:0' if gpu is not None else 'cpu',results,capacity=mode=='capacity')
        passed=True
    finally:write_json(dest/'report.json',dict(passed=passed,checks=results,mode=mode))
