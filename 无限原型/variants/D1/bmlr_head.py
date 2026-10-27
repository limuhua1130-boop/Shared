"""Thin adapter over BYTE-UNCHANGED official HBNN BMLR.py.

Only dtype/device/checkpoint plumbing and diagnostics are supplied here.
The official constructor, direction normalization, exp scale and logits run.
"""
import hashlib
import importlib
from pathlib import Path
import torch
import lib

# Both projects use lib/. Extend its namespace, never replace HEEG lib/lorentz.
VENDOR=Path(__file__).resolve().parent/'hbnn_author/lib'
if str(VENDOR) not in lib.__path__:lib.__path__.append(str(VENDOR))
from lib.bnn.BMLR import BMLR as OfficialBMLR
from lib.bnn.Geometry.base import EPS as OFFICIAL_EPS
if Path(importlib.import_module('lib.bnn.BMLR').__file__).resolve()!=VENDOR/'bnn/BMLR.py':
    raise RuntimeError('Unexpected preloaded HBNN source; require pinned vendor path')
EPS=OFFICIAL_EPS[torch.float64]

def rng_digest():
    return hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()

class BMLR(OfficialBMLR):
    def __init__(self,dim,classes=9,metric='lorentz'):
        # The author uses the ambient default dtype. Limit float64 to head
        # construction; leave all base-model initialization semantics intact.
        previous=torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:super().__init__(n_classes=classes,dim=dim,K=-1.,bias=True,metric=metric)
        finally:torch.set_default_dtype(previous)
        # Official K is an ordinary Tensor; register without changing its value
        # so .to(device), state_dict and strict checkpoint loading include it.
        curvature=self.K;del self.K
        self.register_buffer('K',curvature)
        self.last_audit={}

    def forward(self,x):
        if x.dtype!=torch.float64 or x.shape[-1]!=self.d+(self.metric=='lorentz'):
            raise ValueError('Expected actual float64 BMLR features')
        # Diagnostics never provide a surrogate score or gradient.
        with torch.no_grad():
            norm=self.weight_v.norm(dim=-1,keepdim=True)
            if not bool(torch.isfinite(norm).all() and (norm>=EPS).all()):
                raise FloatingPointError('Degenerate BMLR direction')
            v=self.weight_v/norm.clamp_min(EPS);scale=self.weight_g.exp()
            if self.metric=='lorentz':
                raw=(-self.K).sqrt()*(x[...,:1]-x[...,1:]@v.T)
                valid=raw>0;events=(raw<EPS).sum();minimum=raw.min()
            else:
                q=x.square().sum(-1,keepdim=True)
                numerator=v.square().sum(-1)-2*(-self.K).sqrt()*(x@v.T)+(-self.K)*q
                denominator=1+self.K*q
                valid=(numerator>0)&(denominator>0)
                events=(denominator<EPS).sum();minimum=denominator.min()
            if not bool(valid.all() and torch.isfinite(scale).all() and (scale>0).all()):
                raise FloatingPointError('Invalid BMLR raw domain or positive scale')
            self.last_audit=dict(metric=self.metric,direction_norm_min=float(norm.min()),
                alpha_min=float(scale.min()),alpha_max=float(scale.max()),numeric_guard_count=int(events),
                raw_argument_or_margin_min=float(minimum),raw_valid=True)
        logits=super().forward(x)
        if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite official BMLR logits')
        return logits

def attach_head(model,metric):
    # Do not call feature_dim again: that contains BN updates and dropout.
    dim=model.lmlp.z.shape[-1]
    if dim!=3744 or model.num_classes!=9:raise ValueError('Unexpected head dimensions')
    before=rng_digest();device=next(model.parameters()).device
    model.lmlp=BMLR(dim,9,metric).to(device=device,dtype=torch.float64)
    model.head_initialization=dict(intrinsic_dimension=dim,feature_dimension=dim+(metric=='lorentz'),
        head_parameters=9*(dim+2),RNG_before_head=before,RNG_after_head=rng_digest(),
        initialization='Official constructor with scoped CPU float64; normal(0,d**-.5), log_g=log(norm(weight)), zero bias',
        backbone='Unchanged mother constructor, including discarded old HMLR RNG consumption')
    return model
