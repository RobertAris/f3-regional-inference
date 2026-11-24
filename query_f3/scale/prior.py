"""Published-style bicubic interpolation adjoint as a scale-transfer comparator."""
import torch
from .preview import Preview
def kernel(weight,stride):
    def interp(n):
        r=n//2;out=(r+stride-1)//stride+1
        f=torch.arange(-r,r+1,device=weight.device,dtype=weight.dtype)
        c=torch.arange(-out,out+1,device=weight.device,dtype=weight.dtype)
        t=(c[:,None]-f[None,:]/stride).abs()
        # Keys cardinal cubic interpolation with coefficient -1/2.
        return torch.where(t<=1,1.5*t**3-2.5*t**2+1,
                 torch.where(t<2,-.5*t**3+2.5*t**2-4*t+2,torch.zeros_like(t)))
    return interp(weight.shape[-2]) @ weight @ interp(weight.shape[-1]).T
class CubicPreview(Preview):
    def get(self,layer,stride):
        key=(layer,stride)
        if key not in self.kernels:self.kernels[key]=kernel(layer.weight,stride)
        return self.kernels[key]
