"""Affected numerical checks, executed by03 only."""
import ast
from copy import deepcopy
import torch
import numpy as np
from common import PACKAGE, capture_rng, restore_rng, read_json
from bmlr_head import BMLR,attach_head,EPS
from runtime import seed,config,inputs
from engine import build_model
from optimizer_checkpoint import exact_state_equal

def close(name,a,b,results,atol=2e-10,rtol=2e-9):
    assert torch.isfinite(a).all() and torch.isfinite(b).all(),name
    error=float((a-b).abs().max())
    ok=torch.allclose(a,b,atol=atol,rtol=rtol)
    results.append(dict(check=name,passed=ok,max_abs=error,atol=atol,rtol=rtol))
    assert ok,(name,error)

def algebra(results):
    seed(0)
    dim=3744
    l=BMLR(dim,metric='lorentz');p=BMLR(dim,metric='poincare')
    p.load_state_dict(l.state_dict(),strict=True)
    z=torch.randn(5,dim,dtype=torch.float64)
    z=torch.nn.functional.normalize(z,dim=-1)*torch.tensor([0.,.03,.3,.7,.95],dtype=torch.float64)[:,None]
    # Both expressions are differentiated w.r.t. the SAME ball coordinate.
    zl=z.clone().requires_grad_();zp=z.clone().requires_grad_()
    q=zl.square().sum(-1,keepdim=True)
    h=torch.cat(((1+q)/(1-q),2*zl/(1-q)),dim=-1)
    a=l(h);b=p(zp)
    close('same_point_two_coordinate_logits',a,b,results)
    y=torch.tensor([0,1,2,3,4]);la=torch.nn.functional.cross_entropy(a,y);lb=torch.nn.functional.cross_entropy(b,y)
    ga=torch.autograd.grad(la,(zl,*l.parameters()));gb=torch.autograd.grad(lb,(zp,*p.parameters()))
    for name,x,yv in zip(['input_pullback','weight_v','weight_g','bias'],ga,gb):close('two_coordinate_gradient_'+name,x,yv,results)
    # Independent use of pinned author's actual function bodies, no package imports.
    tree=ast.parse((PACKAGE/'source_evidence/HBNN_BMLR.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    for n in funcs:n.decorator_list=[]
    scope={'torch':torch};exec(compile(ast.Module(body=funcs,type_ignores=[]),'<pinned HBNN functions>','exec'),scope)
    for model,x,name in [(l,h.detach(),'lorentz'),(p,z,'poincare')]:
        v=model.weight_v/model.weight_v.norm(dim=-1,keepdim=True).clamp_min(EPS)
        oracle=scope['_lbusemann_logits' if name=='lorentz' else '_pbusemann_logits'](v.T,x,model.K,model.weight_g.exp(),model.bias,EPS)
        close('official_function_'+name,model(x),oracle,results,1e-13,1e-12)
        from bmlr_head import OfficialBMLR
        previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
        try:raw=OfficialBMLR(n_classes=9,dim=dim,K=-1.,bias=True,metric=name)
        finally:torch.set_default_dtype(previous)
        raw.load_state_dict({k:v for k,v in model.state_dict().items() if k!='K'},strict=True)
        xx=x.clone().requires_grad_();yy=x.clone().requires_grad_()
        wrapped=model(xx);original=raw(yy)
        targets=torch.tensor([0,1,2,3,4])
        gw=torch.autograd.grad(torch.nn.functional.cross_entropy(wrapped,targets),(xx,*model.parameters()))
        go=torch.autograd.grad(torch.nn.functional.cross_entropy(original,targets),(yy,*raw.parameters()))
        for label,w,o in zip(['input','weight_v','weight_g','bias'],gw,go):
            close('official_module_gradient_'+name+'_'+label,w,o,results,1e-13,1e-12)
    # Positive-domain clamp is retained and counted; invalid domain is rejected.
    near=torch.zeros(1,dim,dtype=torch.float64);near[0,0]=1-1e-10
    p(near);assert p.last_audit['numeric_guard_count']==1
    bad=torch.zeros(1,dim,dtype=torch.float64);bad[0,0]=1.001
    try:p(bad)
    except FloatingPointError:pass
    else:raise AssertionError('Outside ball accepted')
    with torch.no_grad():p.weight_v.zero_()
    try:p(z)
    except FloatingPointError:pass
    else:raise AssertionError('Zero direction accepted')
    results.append(dict(check='official_guard_count_and_invalid_domain_direction_rejection',passed=True))

def paired_backbone(cache,results):
    from model import HEEGNet as Mother
    from nets.model import HEEGNet as LorentzMother
    from nets.batchnorm import BatchNormDispersion
    args=dict(chunk_size=7500,num_electrodes=32,num_classes=9,domains=torch.arange(123),
        bnorm_dispersion=BatchNormDispersion.SCALAR,domain_adaptation=True,device='cpu',dtype=torch.float64,**config(0)['model'])
    seed(0);mother=Mother(**args).double();before=deepcopy(mother.state_dict());rng=capture_rng()
    metric=config(0)['head'].get('metric') or ('lorentz' if config(0)['variant']=='D1' else 'poincare')
    attach_head(mother,metric);expected_rng=capture_rng()
    seed(0);actual=build_model(config(0),'cpu')
    assert exact_state_equal(expected_rng,capture_rng())
    assert all(torch.equal(x,actual.state_dict()[k]) for k,x in before.items() if not k.startswith('lmlp.'))
    assert exact_state_equal(mother.state_dict(),actual.state_dict())
    groups=actual.configure_optimizers(lr=.001,weight_decay=.001).param_groups
    for name,param in actual.lmlp.named_parameters():
        group=next(g for g in groups if any(v is param for v in g['params']))
        assert group['weight_decay']==.001,name
    results.append(dict(check='own_mother_all_nonhead_state_exact_and_constructor_plus_head_RNG',passed=True,
        head=actual.head_initialization,optimizer_head_weight_decay=.001))
    if metric=='lorentz':return
    seed(0);oracle=LorentzMother(**args).double();attach_head(oracle,'lorentz')
    assert exact_state_equal(expected_rng,capture_rng())
    bp={n:p for n,p in actual.named_parameters() if p.requires_grad}
    lp={n:p for n,p in oracle.named_parameters() if p.requires_grad}
    assert bp.keys()==lp.keys()
    for n,pv in bp.items():
        if pv.requires_grad:assert torch.equal(pv,lp[n]),n
    # Initial fixed geometric means correspond under the coordinate map.
    bs=actual.state_dict();ls=oracle.state_dict()
    for n,bv in bs.items():
        if n.startswith('bn.') and n in ls and bv.shape!=ls[n].shape:
            lv=ls[n]
            assert bv.shape==(*lv.shape[:-1],lv.shape[-1]-1),n
            close('initial_geometric_buffer_'+n,bv,lv[...,1:]/(lv[...,:1]+1),results,1e-12,1e-10)
    _,rows,splits=inputs(cache);ids=splits[0]['train'][:4]
    array=np.load(cache/'X.npy',mmap_mode='r')
    x=torch.from_numpy(np.array(array[ids],copy=True));d=torch.tensor([rows[i]['domain'] for i in ids])
    y=torch.tensor([rows[i]['label'] for i in ids])
    actual.train();oracle.train();r=capture_rng()
    ob,fb=actual(x,d);restore_rng(r);ol,fl=oracle(x,d)
    close('paired_B4_geometric_features',fb,fl[:,1:]/(fl[:,:1]+1),results,2e-5,2e-5)
    close('paired_B4_BMLR_logits',ob,ol,results,2e-5,2e-5)
    torch.nn.functional.cross_entropy(ob,y).backward();torch.nn.functional.cross_entropy(ol,y).backward()
    errors=[]
    for n,pv in bp.items():
        assert (pv.grad is None)==(lp[n].grad is None),n
        if pv.grad is not None:
            error=float((pv.grad-lp[n].grad).norm());scale=float(lp[n].grad.norm())
            assert torch.isfinite(pv.grad).all() and torch.isfinite(lp[n].grad).all()
            assert error<=2e-4+2e-3*scale,(n,error,scale)
            errors.append(dict(parameter=n,absolute_L2=error,reference_L2=scale))
    results.append(dict(check='paired_B4_active_backbone_and_head_gradients',passed=True,errors=errors,
        scope='Matched initial values and one input; not training trajectory equivalence'))
