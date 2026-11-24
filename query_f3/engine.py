"""Frozen F3 spatial queries and event-conditioned global response summaries."""
from dataclasses import dataclass
import numpy as np
import torch
from .norm_block import apply_block


@dataclass
class Packet:
    """Events in native coordinate order with a stable spatial lookup."""
    xy: np.ndarray
    ev: np.ndarray
    size: tuple
    bucket: int | None = 32

    def __post_init__(self):
        width, height = self.size
        x, y = self.xy.T
        if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
            raise ValueError('event outside sensor')
        if self.bucket is None:
            return
        self.ny = (height+self.bucket-1)//self.bucket
        nx = (width+self.bucket-1)//self.bucket
        ids = (x.astype(np.int64) // self.bucket) * self.ny + y // self.bucket
        self.order = np.argsort(ids, kind="stable")
        self.offsets = np.r_[0, np.bincount(ids, minlength=nx*self.ny).cumsum()]

    def subset(self, rect):
        x, y, w, h = rect
        if self.bucket is None:
            a,b=self.xy.T
            return np.flatnonzero((a>=x)&(a<x+w)&(b>=y)&(b<y+h))
        parts = []
        for i in range(x // self.bucket, (x+w-1) // self.bucket + 1):
            lo = i*self.ny + y//self.bucket
            hi = i*self.ny + (y+h-1)//self.bucket + 1
            parts.append(self.order[self.offsets[lo]:self.offsets[hi]])
        indices = np.concatenate(parts)
        a,b = self.xy[indices].T
        return np.sort(indices[(a>=x)&(a<x+w)&(b>=y)&(b<y+h)])

    def grid(self, enc, rect=None):
        if rect is None:
            x,y,w,h = 0,0,*self.size
            xy,ev = self.xy,self.ev
        else:
            x,y,w,h = rect
            indices = self.subset(rect)
            xy,ev = self.xy[indices],self.ev[indices]
        device = next(enc.parameters()).device
        c = enc.downsample_layers[0][0].in_channels
        grid = next(enc.parameters()).new_zeros((w,h,c))
        if len(ev):
            encoded = enc.multi_hash_encoder(torch.from_numpy(ev).to(grid)[None])[0]
            idx = torch.as_tensor(xy.astype(np.int64) - [x,y],device=device)
            grid.index_put_(tuple(idx.T),encoded,accumulate=True)
        return grid.permute(2,0,1)[None]


def network(enc,head):
    for stem,stage in zip(enc.downsample_layers,enc.stages,strict=True):
        yield stem, list(stage), False
    if head is not None:
        yield None,list(head.decoder),True


def trace(enc,head,grid,*,fixed=None):
    """Shared factors for a single image. Flow's native convolution uses y,x."""
    gains=[]
    x=grid
    field=None
    for stem,blocks,transpose in network(enc,head):
        if transpose:
            field=x
            x=x.transpose(2,3)
        else:
            x=stem(x)
        for block in blocks:
            x,g=apply_block(block,x,fixed=None if fixed is None else fixed[len(gains)])
            gains.append(g)
    if head is None:
        return x,None,gains
    return field,head.pred(x).transpose(2,3),gains


def halo(rect, size, radius):
    x,y,w,h=rect
    x0=max(0,x-radius); y0=max(0,y-radius)
    return x0,y0,min(size[0],x+w+radius)-x0,min(size[1],y+h+radius)-y0


def query(enc, head, packet, rect, gains):
    x,y,w,h=rect
    if min(x,y)<0 or min(w,h)<=0 or x+w>packet.size[0] or y+h>packet.size[1]:
        raise ValueError('ROI must be positive and contained within the sensor')
    r=18 if head is None else 34
    region=halo(rect,packet.size,r)
    field,flow,_=trace(enc,head,packet.grid(enc,region),fixed=gains)
    x,y,w,h=rect
    sx=slice(x-region[0],x-region[0]+w); sy=slice(y-region[1],y-region[1]+h)
    return field[:,:,sx,sy],None if flow is None else flow[:,:,sx,sy]


def clipped_axis(length, radius):
    """Map physical coordinates to a minimum empty-canvas representative."""
    n=min(length,2*radius+1)
    t=np.arange(length)
    return np.where(t<radius,t,np.where(t>=length-radius,t-length+n,radius))


def prototype_index(enc,head,packet):
    """Exact per-window factors from disjoint foreground cores and one empty canvas.

    Each nonempty halo is explicit. Empty halo interiors are represented by
    boundary distance. Factor reductions count each native pixel exactly once.
    """
    width,height=packet.size
    r=18 if head is None else 34
    ax,ay=clipped_axis(width,r),clipped_axis(height,r)
    px,py=int(ax.max())+1,int(ay.max())+1
    blocks=[]
    # Use a single bounding box for compact input. The full-coverage path
    # gives a native dense fallback for dispersed events.
    if len(packet.xy):
        lo=packet.xy.min(0); hi=packet.xy.max(0)
        core=halo((*lo,*(hi-lo+1)),packet.size,r)
        region=halo(core,packet.size,r)
        if region[2]*region[3] > .78*width*height:
            return trace(enc,head,packet.grid(enc))[2], 'dense'
        gx,gy,_,_=region
        x,y,w,h=core
        core_coords=[slice(x-gx,x+w-gx),slice(y-gy,y+h-gy)]
        blocks=[packet.grid(enc,region)]
    xc=np.bincount(ax,minlength=px); yc=np.bincount(ay,minlength=py)
    count=xc[:,None]*yc[None,:]
    if blocks:
        count-=np.bincount(ax[x:x+w],minlength=px)[:,None]*np.bincount(ay[y:y+h],minlength=py)[None,:]
    device=next(enc.parameters()).device
    weights=torch.from_numpy(count.astype('f')).to(device)
    blocks.append(next(enc.parameters()).new_zeros((1,enc.downsample_layers[0][0].in_channels,px,py)))
    gains=[]
    for stem,stage,transpose in network(enc,head):
        if transpose:
            blocks=[z.transpose(2,3) for z in blocks]
            weights=weights.T
            if len(blocks)>1:core_coords.reverse()
        else:
            blocks=[stem(z) for z in blocks]
        for b in stage:
            us=[b.act(b.pwconv1(b.norm(b.dwconv(z).permute(0,2,3,1)))) for z in blocks]
            energy=(us[-1][0].square()*weights[:,:,None]).sum((0,1))
            if len(blocks)>1:
                energy+=us[0][0,core_coords[0],core_coords[1]].square().sum((0,1))
            g=energy.sqrt()
            g=g/(g.mean()+1e-6)
            gains.append(g)
            blocks=[z+b.drop_path(b.pwconv2(b.grn.gamma*(u*g)+b.grn.beta+u).permute(0,3,1,2)) for z,u in zip(blocks,us,strict=True)]
    return gains,'prototype'


def sampled_index(enc,head,packet,k,kind,seed):
    """Stratified cluster sampling with exact area weights and shared stage factors."""
    bw,bh=32,24
    width,height=packet.size
    if width%bw or height%bh:raise ValueError('whole native cores required')
    nx,ny=width//bw,height//bh
    rects=np.indices((nx,ny)).reshape(2,-1).T*np.array([bw,bh])
    rng=np.random.default_rng(seed)
    if kind in ('stratified','prototype_stratified','representative'):
        key=packet.xy[:,0].astype(np.int64)*height+packet.xy[:,1]
        points=np.unique(key)
        ids=(points//height//bw)*ny+(points%height//bh)
        counts=np.bincount(ids,minlength=nx*ny)
        order=np.lexsort((rng.random(nx*ny),counts))
    elif kind=='uniform':
        order=rng.permutation(nx*ny)
    else:raise ValueError(kind)
    strata=np.array_split(order,k)
    chosen=[s[len(s)//2] if kind=='representative' else s[rng.integers(len(s))] for s in strata]
    weights=[len(s) for s in strata]
    centers=rects[chosen]
    r=34 if head is not None else 18
    # Each core is surrounded by a complete conditional receptive field.
    # Physical padding is reset to zero before every depthwise convolution.
    grids=[];masks=[]
    for x,y in centers:
        core=(int(x),int(y),bw,bh)
        xx,yy,ww,hh=halo(core,packet.size,r)
        g=packet.grid(enc,(xx,yy,ww,hh))
        left=xx-x+r;top=yy-y+r
        pads=(int(top),int(bh+2*r-hh-top),int(left),int(bw+2*r-ww-left))
        grids.append(torch.nn.functional.pad(g,pads))
        masks.append(torch.nn.functional.pad(torch.ones_like(g[:,:1]),pads))
    x=torch.cat(grids);mask=torch.cat(masks)
    wts=x.new_tensor(weights)
    bounds=(r,r+bw,r,r+bh)
    gs=[]
    control = kind == 'prototype_stratified'
    if control:
        ax,ay=clipped_axis(width,r),clipped_axis(height,r)
        px,py=int(ax.max())+1,int(ay.max())+1
        count=np.bincount(ax,minlength=px)[:,None]*np.bincount(ay,minlength=py)[None,:]
        for (cx,cy),wt in zip(centers,weights,strict=True):
            count-=wt*np.bincount(ax[cx:cx+bw],minlength=px)[:,None]*np.bincount(ay[cy:cy+bh],minlength=py)[None,:]
        correction=x.new_tensor(count)
        blank=x.new_zeros((1,x.shape[1],px,py))
    for stem,stage,transpose in network(enc,head):
        if transpose:
            x=x.transpose(2,3);mask=mask.transpose(2,3)
            bounds=(r,r+bh,r,r+bw)
            if control: blank=blank.transpose(2,3);correction=correction.T
        else:
            x=stem(x)
            if control: blank=stem(blank)
        for b in stage:
            if control:
                x=x*mask
                def pre(z):
                    return b.act(b.pwconv1(b.norm(b.dwconv(z).permute(0,2,3,1))))
                u=pre(x);v=pre(blank)
                a0,a1,b0,b1=bounds
                part=u[:,a0:a1,b0:b1].square().sum((1,2))
                energy=(part*wts[:,None]).sum(0)+(v[0].square()*correction[:,:,None]).sum((0,1))
                norm=energy.clamp_min(0).sqrt()
                g=norm/(norm.mean()+1e-6)
                def post(z,a):
                    return z+b.drop_path(b.pwconv2(b.grn.gamma*(a*g)+b.grn.beta+a).permute(0,3,1,2))
                x=post(x,u);blank=post(blank,v)
            else:
                x,g=apply_block(b,x*mask,bounds=bounds,weights=wts)
            gs.append(g)
    return gs
