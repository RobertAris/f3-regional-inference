"""Estimate late global response factors from measured native prefix statistics."""
import numpy as np
import torch
from torch.nn import functional as F
from .engine import apply_block,network,halo

def prefix(enc,grid,depths=(1,2,3)):
    gains=[];found={}
    x=grid
    for stem,stage in zip(enc.downsample_layers,enc.stages,strict=True):
        x=stem(x)
        for b in stage:
            x,g=apply_block(b,x);gains.append(g)
            if len(gains) in depths:found[len(gains)]=(x,torch.stack(gains))
            if len(gains)>=max(depths):return found
    raise ValueError(depths)

def suffix_query(enc,head,packet,rect,k,state,gains):
    x,y,w,h=rect
    margin=34-3*k
    if min(x,y)<0 or min(w,h)<=0 or x+w>packet.size[0] or y+h>packet.size[1]:raise ValueError('contained ROI required')
    a,b,c,d=halo(rect,packet.size,margin)
    v=state[:,:,a:a+c,b:b+d];i=0;field=None
    for stem,stage,transpose in network(enc,head):
        if transpose:
            field=v;v=v.transpose(2,3)
        elif i>=k:v=stem(v)
        for block in stage:
            if i>=k:v,_=apply_block(block,v,fixed=None if gains is None else gains[i])
            i+=1
    out=head.pred(v).transpose(2,3)
    ix=slice(x-a,x-a+w);iy=slice(y-b,y-b+h)
    return field[:,:,ix,iy],out[:,:,ix,iy]

def moments(grid):
    """Scale-wise moments and mean-removed energy of the native scatter."""
    x=grid;out=[]
    for level in range(5):
        z=x[0].reshape(x.shape[1],-1)
        mu=z.mean(1)
        raw=(z@z.T)/z.shape[1]
        out.extend((mu,raw.flatten()))
        if level!=4:x=F.avg_pool2d(x,kernel_size=2,stride=2)
    return torch.cat(out)
class Readout:
    def __init__(self,x,y,reg=.15):
        x=x.astype(np.float64)
        self.mean=x.mean(0)
        std=x.std(0)
        self.scale=np.maximum(std,np.max(std)*.015+1e-10)
        z=(x-self.mean)/self.scale
        flat=y.reshape(len(y),-1)
        self.center=flat.mean(0)
        # Penalize relative to standardized descriptor width.
        dual=np.linalg.solve(z@z.T+np.eye(len(z))*reg*z.shape[1],flat-self.center)
        layout=y.shape[1:]
        self.meta=[torch.tensor(v,dtype=torch.float32,device='mps') for v in (self.mean,self.scale,z.T,dual,self.center)]
        self.layout=layout
    def __call__(self,data):
        mu,s,basis,dual,center=self.meta
        return (center+(((data-mu)/s)@basis)@dual).reshape(self.layout).clamp_min(0)

def specs():
    return ['moment','moment_thumb','prefix1','prefix2','prefix3','prefix1_thumb','prefix2_thumb','prefix3_thumb']
def desc(mode,mom,gs,thumb):
    if mode.startswith('moment'):items=[mom]
    else:items=[gs[:int(mode[6])].reshape(-1)]
    if mode.endswith('_thumb'):items.append(thumb.reshape(-1))
    return torch.cat(items)

class Context:
    def __init__(self,depth,state,gains):self.depth=depth;self.state=state;self.gains=gains
    def query(self,enc,head,packet,rect):
        return suffix_query(enc,head,packet,rect,self.depth,self.state,self.gains)
