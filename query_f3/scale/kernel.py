"""Project depthwise kernels onto a bilinearly interpolated coarse grid."""

from contextlib import nullcontext
from numbers import Integral

import torch


def _tent_matrix(radius: int, stride: int, weight: torch.Tensor) -> torch.Tensor:
    """Return interpolation weights indexed by coarse node and fine offset."""
    coarse_radius = (radius + stride - 1) // stride
    if coarse_radius == 1:
        # Python integer distances also handle strides larger than int64.
        offsets = range(-radius, radius + 1)
        return weight.new_tensor([
            [max(stride - abs(offset - center), 0) / stride for offset in offsets]
            for center in (-stride, 0, stride)
        ])

    fine = torch.arange(-radius, radius + 1, device=weight.device)
    coarse = torch.arange(-coarse_radius, coarse_radius + 1, device=weight.device)
    distance = (fine[None, :] - stride * coarse[:, None]).abs()
    return (stride - distance).clamp_min(0).to(weight.dtype) / stride


def projected_kernel(weight: torch.Tensor, sx: int, sy: int) -> torch.Tensor:
    """Return the coarse-grid kernel for a depthwise weight in x, y axis order.

    The result reproduces the native response at coarse nodes wherever the
    native input is the bilinear interpolation of the coarse input throughout
    the kernel footprint, up to floating-point rounding. The caller's boundary
    convention determines where this condition holds.
    """
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("weight must be a floating-point torch.Tensor")
    if weight.layout != torch.strided or weight.ndim != 4:
        raise ValueError("weight must be dense with shape [C, 1, Kx, Ky]")
    channels, inputs_per_group, kx, ky = weight.shape
    if channels < 1 or inputs_per_group != 1:
        raise ValueError("weight must have positive C and one input per group")
    if kx % 2 == 0 or ky % 2 == 0:
        raise ValueError("Kx and Ky must be positive and odd")
    if any(isinstance(s, bool) or not isinstance(s, Integral) for s in (sx, sy)):
        raise TypeError("sx and sy must be integers")
    if sx < 1 or sy < 1:
        raise ValueError("sx and sy must be positive")
    sx, sy = int(sx), int(sy)

    project_x = sx != 1 and kx != 1
    project_y = sy != 1 and ky != 1
    if not project_x and not project_y:
        return weight

    # Accumulate in at least float32, then restore the input dtype.
    work_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    projected = weight.to(work_dtype)
    device_type = weight.device.type
    autocast = nullcontext()
    if torch.amp.autocast_mode.is_autocast_available(device_type):
        autocast = torch.autocast(device_type=device_type, enabled=False)
    with autocast:
        if project_x:
            projected = _tent_matrix(kx // 2, sx, projected) @ projected
        if project_y:
            projected = projected @ _tent_matrix(ky // 2, sy, projected).T
    return projected.to(weight.dtype)
