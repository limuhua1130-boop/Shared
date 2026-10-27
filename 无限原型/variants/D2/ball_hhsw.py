"""Frozen HEEG reference law and HHSW in native ball coordinates."""
import numpy as np
import torch
from torch.nn import functional as F
from hsssw.lib.hhsw import emd1D

def reference(x,manifold):
    # Original samples D+1, normalizes all coordinates; its projx then discards
    # the time component after expmap0. Preserve this effective distribution.
    g=torch.normal(0,1,size=(*x.shape[:-1],x.shape[-1]+1),dtype=x.dtype,device=x.device)
    g=g/(g.norm(dim=-1,keepdim=True)+1e-8)
    n=(g[...,1:].square().sum(-1,keepdim=True)-g[...,:1].square()).clamp_min(1e-8).sqrt()
    spatial=torch.sinh(n.clamp_max(10))*g[...,1:]/n
    return manifold.from_spatial(spatial)

def busemann(x,p):
    q=x.square().sum(-1,keepdim=True)
    return torch.log((1+q-2*x@p.T)/(1-q))

def cost(x,y,p):return emd1D(busemann(x,p).T,busemann(y,p).T,p=2).mean()

def swd_loss(x,domains,manifold):
    total=0
    for d in domains.unique():
        a=x[domains==d];b=reference(a,manifold)
        p=F.normalize(torch.from_numpy(np.random.normal(size=(1000,x.shape[-1]))),dim=-1).to(x)
        total=total+cost(a,b,p)
    return total
