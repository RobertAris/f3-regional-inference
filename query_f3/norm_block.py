"""Window-wide response normalization for the frozen ConvNeXtV2 block."""

import torch


@torch.no_grad()
def apply_block(block, x, *, bounds=None, weights=None, fixed=None):
    """Apply an eval block to BCHW tiles belonging to one window.

    The shared bounds (x0, x1, y0, y1) slice the first and second spatial
    axes respectively. With no bounds, the full tiles contribute. Weights
    are a same-device vector of nonnegative tile multiplicities, defaulting
    to one per tile. The returned factor has one value per hidden channel.
    Pass it as fixed to reuse it directly and skip the spatial reductions.
    The residual update covers the full tiles, including their halos.
    """
    u = block.dwconv(x).permute(0, 2, 3, 1)
    u = block.act(block.pwconv1(block.norm(u)))

    g = fixed
    if g is None:
        interior = u
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            interior = u[:, x0:x1, y0:y1, :]

        if weights is None:
            norms = torch.norm(interior, p=2, dim=(0, 1, 2))
        else:
            # Square and accumulate low-precision activations in float32.
            dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
            tile_energy = interior.to(dtype=dtype).square().sum(dim=(1, 2))
            multiplicities = weights.to(dtype=dtype)[:, None]
            norms = (tile_energy * multiplicities).sum(dim=0).sqrt()

        g = (norms / (norms.mean() + 1e-6)).to(dtype=u.dtype)

    response = block.grn.gamma * (u * g) + block.grn.beta + u
    update = block.pwconv2(response).permute(0, 3, 1, 2)
    return x + block.drop_path(update), g
