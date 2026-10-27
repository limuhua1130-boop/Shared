"""Native unit Poincare ball formulas (sectional curvature -1).

Algebraic counterparts of frozen HEEG operators; no Lorentz layer/forward calls.
Frechet implicit derivative is the author's existing Ball implementation.
"""
import torch
from torch import nn
from Geometry.hyperbolic.frechetmean.backward.ball_backward import frechet_ball_backward
from Geometry.hyperbolic.utils.utils_common import darcosh, arcosh

class BallMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, K):
        w=torch.ones(x.shape[:-1],dtype=x.dtype,device=x.device)
        mu=x[0].clone(); x2=x.square().sum(-1); den=1-x2
        for _ in range(100):
            old=mu; y=(x-mu).square().sum(-1)/(den*(1-mu.square().sum()))
            # 2*darcosh(1+2y) is author's l_prime(y), with its analytic y=0 limit.
            alpha=2*darcosh(1+2*y)/den
            a=alpha.sum(); c=(alpha*x2).sum(); b=(alpha[:,None]*x).sum(0)
            # Rationalization of (a+c-sqrt((a+c)^2-4||b||²))/(2||b||²).
            # Finite at b=0; exact same fixed-point map, no new mean objective.
            disc=((a+c).square()-4*b.square().sum()).clamp_min(0)
            mu=2*b/(a+c+disc.sqrt())
            delta=(mu-old).norm()
            if bool(delta<1e-12) or bool(delta<1e-12*old.norm()):break
        ctx.save_for_backward(x,mu,w,K)
        return mu
    @staticmethod
    def backward(ctx, grad):
        x,mu,w,K=ctx.saved_tensors
        dx,_,_=frechet_ball_backward(x,mu,grad,w,K)
        return dx,None

class UnitBall(nn.Module):
    def __init__(self,parameter=True):
        super().__init__()
        if parameter:self.k=nn.Parameter(torch.tensor(1.),requires_grad=False)
        else:self.register_buffer('k',torch.tensor(1.))
        self.projection_events=0
    def project(self,x):
        n=x.norm(dim=-1,keepdim=True); maximum=1-1e-8
        self.projection_events+=int((n.detach()>maximum).sum())
        return x*(maximum/n.clamp_min(maximum)).clamp_max(1)
    def from_spatial(self,s):
        return self.project(s/(1+torch.sqrt(1+s.square().sum(-1,keepdim=True))))
    def spatial(self,x):return 2*x/(1-x.square().sum(-1,keepdim=True))
    def dist0(self,x):return 2*torch.atanh(x.norm(dim=-1))
    def dist(self,x,y):
        # Preserve original Geometry.Hyperboloid's arcosh floor for BN variance.
        q=1+2*(x-y).square().sum(-1)/((1-x.square().sum(-1))*(1-y.square().sum(-1)))
        return arcosh(q)
    def add(self,x,y):
        x2=x.square().sum(-1,keepdim=True);y2=y.square().sum(-1,keepdim=True);xy=(x*y).sum(-1,keepdim=True)
        return self.project(((1+2*xy+y2)*x+(1-x2)*y)/(1+2*xy+x2*y2))
    def scale(self,x,r):
        n=x.norm(dim=-1,keepdim=True);safe=n.clamp_min(1e-15)
        factor=torch.where(n>1e-15,torch.tanh(r*torch.atanh(n))/safe,torch.ones_like(n)*r)
        return self.project(factor*x)
    def geodesic(self,x,y,t):return self.add(x,self.scale(self.add(-x,y),t))
    def centroid(self,x):
        q=x.square().sum(-1,keepdim=True);den=1-q
        a=((1+q)/den).mean(-2);b=(2*x/den).mean(-2)
        norm=(a.square()-b.square().sum(-1,keepdim=True)).abs().clamp_min(1e-8).sqrt()
        return self.project(b/(a+norm))
    def flatten(self,x):return self.from_spatial(self.spatial(x).flatten(1))
    def activation(self,x):return self.from_spatial(torch.nn.functional.elu(self.spatial(x)))

class BallLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, bias=False):
        super().__init__();self.manifold=manifold
        self.weight=nn.Linear(in_features,out_features,bias=bias)
        nn.init.uniform_(self.weight.weight,-.02,.02)
        if bias:nn.init.constant_(self.weight.bias,0)
        self.scale=nn.Parameter(torch.ones(()),requires_grad=False)
    def forward(self,x):
        q=x.square().sum(-1,keepdim=True);den=1-q
        # Keep the original inactive output row in the parameter tensor.
        s=torch.nn.functional.linear(2*x/den,self.weight.weight[1:,1:])
        s=s+(1+q)/den*self.weight.weight[1:,0]
        if self.weight.bias is not None:s=s+self.weight.bias[1:]
        return self.manifold.from_spatial(s)

class BallConv2d(nn.Module):
    def __init__(self,manifold,in_channels,out_channels,kernel_size,padding=(0,0),bias=False,stride=1):
        super().__init__();assert kernel_size==1 and padding==(0,0) and stride==1
        self.manifold=manifold
        self.linearized_kernel=BallLinear(manifold,in_channels,out_channels,bias)
        # Exact frozen lc1 initialization including the unused row.
        bound=(2/(in_channels-1))**.5
        nn.init.uniform_(self.linearized_kernel.weight.weight,-bound,bound)
    def forward(self,x):return self.linearized_kernel(x)

class BallELU(nn.Module):
    def __init__(self,manifold):super().__init__();self.manifold=manifold
    def forward(self,x):return self.manifold.activation(x)

class BallAvgPool2d(nn.Module):
    def __init__(self,manifold,kernel_size,stride):
        super().__init__();assert kernel_size==(1,8) and stride==8;self.manifold=manifold
    def forward(self,x):
        b,h,w,d=x.shape;assert h==1
        # Same disjoint windows; vectorized arithmetic only, no validation splitting.
        return self.manifold.centroid(x[:,:,:w//8*8].reshape(b,1,w//8,8,d))

class BallMLR(nn.Module):
    def __init__(self,manifold,num_features,num_classes):
        super().__init__();self.manifold=manifold
        self.a=nn.Parameter(torch.zeros(num_classes))
        self.z=nn.Parameter(torch.nn.functional.pad(torch.zeros(num_classes,num_features-2),(1,0),value=1))
        bound=1/self.z.shape[1]**.5
        nn.init.uniform_(self.z,-bound,bound);nn.init.uniform_(self.a,-bound,bound)
    def forward(self,x):
        q=x.square().sum(-1,keepdim=True);den=1-q;n=self.z.norm(dim=-1)
        wt=torch.sinh(self.a)*n;ws=torch.cosh(self.a[:,None])*self.z
        beta=(-wt.square()+ws.norm(dim=-1).square()).sqrt()
        alpha=-(1+q)/den*wt+torch.cosh(self.a)*torch.inner(2*x/den,self.z)
        return beta*torch.asinh(alpha/beta)
