import torch
from torch import nn
from nets.batchnorm import SchedulableBatchNorm, BaseDomainBatchNorm, BatchNormTestStatsMode, BatchNormDispersion
from ball import UnitBall, BallMean

class BallDomainStats(SchedulableBatchNorm):
    def __init__(self,dim,std,eta=1.,eta_test=.1):
        super().__init__(eta,eta_test)
        self.manifold=UnitBall(parameter=False);self.register_buffer('K',torch.tensor(-1.,dtype=torch.float64))
        self.dispersion=BatchNormDispersion.SCALAR;self.eps=1e-5
        mean=torch.zeros(dim);var=torch.ones(1)
        self.register_buffer('running_mean',mean);self.register_buffer('running_mean_test',mean)
        self.register_buffer('running_var',var);self.register_buffer('running_var_test',var)
        self.std=std  # frozen author created this shared parameter but never used it.
    @torch.no_grad()
    def initrunningstats(self,x):
        self.running_mean=BallMean.apply(x,self.K)
        self.running_mean_test=self.running_mean.clone()
        self.running_var=self.manifold.dist(x,self.running_mean).square().mean()
        self.running_var_test=self.running_var.clone()
    def forward(self,X):
        shape=X.shape;x=X.reshape(-1,shape[-1]);m=self.manifold
        if self.training:
            bm=BallMean.apply(x,self.K)
            rm=m.geodesic(self.running_mean,bm,self.eta)
            bv=m.dist(x,bm).square().mean()
            rv=(1-self.eta)*self.running_var+self.eta*bv
        else:
            if self.test_stats_mode==BatchNormTestStatsMode.REFIT:self.initrunningstats(x)
            rm=self.running_mean_test;rv=self.running_var_test
        out=m.scale(m.add(-rm,x),1/(rv+self.eps).sqrt())
        if self.training:
            with torch.no_grad():
                self.running_mean=rm.clone()
                self.running_mean_test=m.geodesic(self.running_mean_test,bm,self.eta_test)
                self.running_var=rv.clone()
                self.running_var_test=(1-self.eta_test)*self.running_var_test+self.eta_test*m.dist(x,bm).square().mean()
        return out.reshape(shape)

class BallDomainBN(BaseDomainBatchNorm):
    def __init__(self,shape,batchdim,domains,learn_mean,learn_std,dispersion,eta,eta_test,**kwargs):
        super().__init__();assert not learn_mean and learn_std and dispersion==BatchNormDispersion.SCALAR
        self.std=nn.Parameter(torch.ones(1))
        self.dispersion=dispersion
        for d in domains:self.add_domain_(BallDomainStats(shape[-1],self.std,eta,eta_test),d)
        self.set_test_stats_mode(BatchNormTestStatsMode.BUFFER)
