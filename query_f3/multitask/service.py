"""Training-free native-resolution queries for released F3 perception heads."""

import numpy as np
import torch
from torch.nn import functional as F

from query_f3.depth_context import prefix
from query_f3.engine import Packet, halo
from query_f3.scale.prior import CubicPreview
from query_f3.scale.preview import Preview
from query_f3.quadrature.batch import batched_suffix
from .models import dense_output


def packet(events, size, table=None):
    if table is not None:
        from .lookup import LookupPacket

        return LookupPacket(events, size, table)
    xy = events[:, 1:3].astype(np.int64)
    age = (-1 - events[:, 0].astype(np.int64)) // 1000
    coords = np.column_stack((xy / np.asarray(size), age / 20)).astype("f")
    return Packet(xy, coords, size, None)


def active_regions(events, size, count=4, extent=48):
    """Choose the busiest site in each spatial cell without looking at outputs."""
    w, h = size
    if not 1 <= extent <= min(size) or not 1 <= count <= w * h:
        raise ValueError("Request does not fit sensor")
    cols = min(count, max(1, round(np.sqrt(count * w / h))))
    rows = int(np.ceil(count / cols))
    bins = np.bincount(events[:, 2].astype("int64") * w + events[:, 1], minlength=w * h).reshape(h, w)
    result = []
    for n in range(count):
        j, i = divmod(n, cols)
        x0, x1 = i * w // cols, (i + 1) * w // cols
        y0, y1 = j * h // rows, (j + 1) * h // rows
        values = bins[y0:y1, x0:x1]
        if values.max():
            dy, dx = np.unravel_index(values.argmax(), values.shape)
        else:
            dy, dx = (y1 - y0) // 2, (x1 - x0) // 2
        x = int(np.clip(x0 + dx - extent // 2, 0, w - extent))
        y = int(np.clip(y0 + dy - extent // 2, 0, h - extent))
        result.append((x, y, extent, extent))
    return result


def encoder_field(enc, grid):
    for stem, stage in zip(enc.downsample_layers, enc.stages, strict=True):
        grid = stage(stem(grid))
    return grid


def gather_xy(field, boxes):
    return torch.cat([field[:, :, x : x + w, y : y + h] for x, y, w, h in boxes])


def gather_yx(field, boxes):
    return [field[:, :, y : y + h, x : x + w] for x, y, w, h in boxes]


def crop_field(enc, p, boxes):
    out = []
    for box in boxes:
        x, y, w, h = box
        a, b, c, d = halo(box, p.size, 18)
        field = encoder_field(enc, p.grid(enc, (a, b, c, d)))
        out.append(field[:, :, x - a : x - a + w, y - b : y - b + h])
    return torch.cat(out)


class FeatureService:
    def __init__(self):
        self.projected = CubicPreview()
        self.plain = Preview()

    def context(self, enc, grid, mode, split, stride):
        from query_f3.depth_context import Context

        st, g = prefix(enc, grid, (split,))[split]
        if split == 6:
            return Context(split, st, g)
        if mode == "oracle":
            from query_f3.engine import trace

            _, _, gains = trace(enc, None, grid)
            return Context(split, st, gains)
        projected = mode in ("regional", "merged")
        if not projected and mode != "plain":
            raise ValueError(mode)
        preview = self.projected if projected else self.plain
        return preview.context(enc, None, st, g, stride, projected, projected)

    def query(self, enc, p, boxes, mode, *, split=3, stride=2, low=None):
        if mode == "crop":
            return crop_field(enc, p, boxes)
        if mode == "released" and len(p.ev):
            if low is not None:
                raise ValueError("Released event hashing requires full precision")
            enc.w, enc.h = p.size
            ev = torch.from_numpy(p.ev).to(next(enc.parameters()).device)
            field = enc(ev, torch.tensor([len(ev)], device=ev.device)).permute(0, 3, 1, 2)
            return gather_xy(field, boxes)
        grid = p.grid(enc)
        if low is not None:
            enc, grid = low, grid.half()
        if mode in ("dense", "released"):
            return gather_xy(encoder_field(enc, grid), boxes)
        if mode == "ordinary":
            coarse = encoder_field(enc, F.avg_pool2d(grid, stride, stride, ceil_mode=True))
            return gather_xy(
                F.interpolate(coarse, size=p.size, mode="bilinear", align_corners=False), boxes
            )
        ctx = self.context(enc, grid, mode, split, stride)
        if ctx.depth == 6:
            return gather_xy(ctx.state, boxes)
        return merged_suffix(enc, ctx, boxes) if mode == "merged" else field_tiles(enc, ctx, boxes)


def global_regions(task, head, field_yx, boxes, mode):
    if mode == "exact":
        from .decoders import depth_context, depth_regions, seg_context, seg_regions

        context_fn, region_fn = (
            (depth_context, depth_regions) if task == "depth" else (seg_context, seg_regions)
        )
        return region_fn(head, context_fn(head, field_yx), boxes)
    if mode == "dense":
        return gather_yx(dense_output(task, head, field_yx), boxes)
    if mode == "ordinary":
        # Standard reduced head input. Return scores on the original sensor grid.
        coarse = F.avg_pool2d(field_yx, 2, 2, ceil_mode=True)
        v = dense_output(task, head, coarse, depth_height=266)
        v = F.interpolate(v, size=field_yx.shape[-2:], mode="bilinear", align_corners=task == "depth")
        return gather_yx(v, boxes)
    if mode == "crop":
        out = []
        for box in boxes:
            x, y, w, h = box
            a, b, c, d = halo(box, field_yx.shape[-2:][::-1], 32)
            whole = dense_output(task, head, field_yx[..., b : b + d, a : a + c])
            out.append(whole[..., y - b : y - b + h, x - a : x - a + w])
        return out
    raise ValueError(mode)


def hybrid_field(enc, p, boxes, split=3, stride=2, preview=None, projected=True):
    """Fill global attention context on a coarse lattice, then replace requested sites."""
    st, g = prefix(enc, p.grid(enc), (split,))[split]
    preview = CubicPreview() if preview is None else preview
    ctx, lattice = preview.context(enc, None, st, g, stride, projected, projected, return_lattice=True)
    # Lattice sample i is native location phase + i*stride, even on odd canvases.
    w, h = p.size
    field = lattice
    for axis, length in ((2, w), (3, h)):
        pos = (torch.arange(length, device=st.device, dtype=st.dtype) - stride // 2) / stride
        pos = pos.clamp(0, field.shape[axis] - 1)
        lo = pos.long()
        hi = (lo + 1).clamp_max(field.shape[axis] - 1)
        shape = [1] * 4
        shape[axis] = length
        field = torch.lerp(
            field.index_select(axis, lo), field.index_select(axis, hi), (pos - lo).reshape(shape)
        )
    native = batched_suffix(enc, None, ctx, boxes)[0]
    for crop, (x, y, w, h) in zip(native, boxes, strict=True):
        field[0, :, x : x + w, y : y + h] = crop
    return field


def merged_suffix(enc, ctx, boxes):
    """Evaluate overlapping equal-size requests together when their bounding tile is cheaper."""
    x0 = min(x for x, y, w, h in boxes)
    y0 = min(y for x, y, w, h in boxes)
    x1 = max(x + w for x, y, w, h in boxes)
    y1 = max(y + h for x, y, w, h in boxes)
    r = 3 * (6 - ctx.depth)
    total = sum((w + 2 * r) * (h + 2 * r) for x, y, w, h in boxes)
    if len(boxes) > 1 and (x1 - x0 + 2 * r) * (y1 - y0 + 2 * r) < total:
        tile = batched_suffix(enc, None, ctx, [(x0, y0, x1 - x0, y1 - y0)])[0]
        return gather_xy(tile, [(x - x0, y - y0, w, h) for x, y, w, h in boxes])
    return field_tiles(enc, ctx, boxes)


def field_tiles(enc, ctx, boxes):
    # Bound a single expanded activation at 64 MiB; complete outputs retain query order.
    w, h = boxes[0][2:]
    r = 3 * (6 - ctx.depth)
    expanded_channels = enc.stages[-1][-1].pwconv1.out_features
    bytes_per_tile = (w + 2 * r) * (h + 2 * r) * expanded_channels * ctx.state.element_size()
    chunk = max(1, (64 * 1024**2) // bytes_per_tile)
    if len(boxes) <= chunk:
        return batched_suffix(enc, None, ctx, boxes)[0]
    return torch.cat(
        [batched_suffix(enc, None, ctx, boxes[i : i + chunk])[0] for i in range(0, len(boxes), chunk)]
    )
