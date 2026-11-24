"""Estimate shared GRNs on a coarse lattice with a physically resampled filter."""
import torch
from torch.nn import functional as F
from query_f3.engine import network
from query_f3.depth_context import prefix,Context
from query_f3.quadrature.batch import batched_suffix
from .kernel import projected_kernel

class Preview:
    def __init__(self):
        self.kernels={}
    def get(self,layer,stride):
        key=(layer,stride)
        if key not in self.kernels:self.kernels[key]=projected_kernel(layer.weight,stride,stride)
        return self.kernels[key]
    def context(self,enc,head,state,observed,stride,first_native,project,*,return_lattice=False):
        depth=len(observed);phase=stride//2
        g=list(observed);i=0;full=state
        x=state[:,:,phase::stride,phase::stride].contiguous()
        for stem,stage,trans in network(enc,head):
            if trans:
                x=x.transpose(2,3);full=full.transpose(2,3) if full is not None else None
            elif i>=depth:
                x=stem(x)
                if i==depth and first_native:full=stem(full)
            for b in stage:
                if i>=depth:
                    conv=b.dwconv
                    if i==depth and first_native:
                        r=conv.kernel_size[0]//2
                        # Stock convolution at centers (phase + stride * index).
                        inp=F.pad(full,(r-phase,r,r-phase,r))
                        z=F.conv2d(inp,conv.weight,conv.bias,stride=stride,groups=conv.groups)
                    elif project:
                        w=self.get(conv,stride);rad=(w.shape[2]//2,w.shape[3]//2)
                        z=F.conv2d(x,w,conv.bias,padding=rad,groups=conv.groups)
                    else:z=conv(x)
                    full=None
                    u=b.act(b.pwconv1(b.norm(z.permute(0,2,3,1))))
                    norms=torch.norm(u,p=2,dim=(0,1,2))
                    gain=norms/(norms.mean()+1e-6)
                    g.append(gain)
                    v=b.grn.gamma*(u*gain)+b.grn.beta+u
                    x=x+b.drop_path(b.pwconv2(v).permute(0,3,1,2))
                i+=1
        ctx=Context(depth,state,torch.stack(g))
        return (ctx,x) if return_lattice else ctx
    def predict(self,enc,head,p,rects,mode):
        k,s,nat,proj=map(int,mode.split('/'))
        st,g=prefix(enc,p.grid(enc),(k,))[k]
        ctx=self.context(enc,head,st,g,s,bool(nat),bool(proj))
        return batched_suffix(enc,head,ctx,rects)
