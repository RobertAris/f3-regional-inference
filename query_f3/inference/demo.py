"""Run the fixed regional preview with two frozen upstream F3 checkpoints."""
import argparse,importlib.util
from pathlib import Path
import numpy as np,torch
from query_f3.engine import Packet
from query_f3.scale.preview import Preview
from query_f3.scale.prior import CubicPreview
from query_f3.quadrature.batch import batched_suffix
from query_f3.depth_context import prefix

def import_file(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('f3-repo','encoder-weights','head-weights','head-config','events','rectangles','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--size',nargs=2,type=int,default=(1280,720),metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    parser.add_argument('--projection',default='bicubic',choices=['bicubic','bilinear'],
                        help='coarse preview interpolation (default: bicubic, as in the main results)')
    opt=parser.parse_args();w,h=opt.size
    src=Path(__file__).resolve().parent
    adapters=src/'loaders'
    if not adapters.is_dir():
        a=src.parents[1]/'archive/binary-occupancy/project';encmod=a/'diagnostics/published_f3.py';headmod=a/'occupancy/released_flow.py'
    else:encmod=adapters/'published_f3.py';headmod=adapters/'released_flow.py'
    enc=import_file(encmod).load_published_f3(opt.f3_repo,opt.encoder_weights,opt.device).eval()
    enc.return_logits=False;enc.return_feat=True;enc.w=w;enc.h=h;enc.frame_sizes=[w,h,20]
    head=import_file(headmod).load_head(opt.f3_repo,opt.head_weights,opt.head_config,opt.encoder_weights,opt.device).eval()
    raw=np.load(opt.events,allow_pickle=False);rs=np.load(opt.rectangles,allow_pickle=False)
    if raw.ndim!=2 or raw.shape[1]!=3 or not np.issubdtype(raw.dtype,np.integer):raise ValueError('events must be integer [relative_us,x,y] rows')
    if np.any(raw[:,0]<-20000) or np.any(raw[:,0]>=0) or np.any(np.diff(raw[:,0].astype('int64'))<0):raise ValueError('events outside sorted half-open 20-ms window')
    if rs.ndim!=2 or rs.shape[1]!=4 or not np.issubdtype(rs.dtype,np.integer):raise ValueError('rectangles must be integer [x,y,width,height] rows')
    xy=raw[:,1:3].astype(np.int64);age=(-1-raw[:,0].astype(np.int64))//1000
    p=Packet(xy,np.column_stack((xy/np.array([w,h]),age/20)).astype('f'),(w,h),None)
    st,g=prefix(enc,p.grid(enc),(5,))[5]
    preview=CubicPreview() if opt.projection=='bicubic' else Preview()
    ctx=preview.context(enc,head,st,g,2,True,True)
    field,vec=batched_suffix(enc,head,ctx,rs)
    np.savez_compressed(opt.output,features=field.cpu().numpy(),flow100ms=vec.cpu().numpy(),rectangles=rs)
if __name__=='__main__':main()
