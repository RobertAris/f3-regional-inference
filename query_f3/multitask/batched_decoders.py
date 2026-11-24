"""Batched regional decoder tails for the full-image F3 contexts."""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoders import (
    DepthContext,
    SegContext,
    _Rect,
    _Size,
    _axis_samples,
    _validated_rects,
)
from .fusion import FoldedSegformer


class _TilePlan(NamedTuple):
    rows: Tensor
    cols: Tensor
    native_rows: Tensor
    native_cols: Tensor


def _groups(rectangles: list[_Rect]) -> Iterable[list[_Rect]]:
    groups: dict[_Size, list[_Rect]] = {}
    for rect in dict.fromkeys(rectangles):
        groups.setdefault(rect[2:], []).append(rect)
    return groups.values()


def _resize_axis(positions: Tensor, source: int, target: int, align_corners: bool) -> Tensor:
    if align_corners:
        return positions * ((source - 1) / max(target - 1, 1))
    return (positions + 0.5) * (source / target) - 0.5


def _plan_axis(
    starts: Sequence[int],
    length: int,
    lattice: int,
    native: int,
    align_corners: bool,
    halo: int,
) -> tuple[Tensor, Tensor, int]:
    # Only the endpoints are needed for the integer bounds of a monotone resize.
    bounds = []
    for start in starts:
        taps = _axis_samples((start, start + length - 1), lattice, native, align_corners)
        bounds.append((taps.lower[0], taps.upper[-1] + 1))
    extent = max(stop - start for start, stop in bounds)

    # Keep every uniformly sized tile inside the lattice before adding the halo.
    origins = torch.tensor(
        [min(start, lattice - extent) for start, _ in bounds],
        dtype=torch.float32,
        device="cpu",
    )[:, None]
    native_pixels = torch.tensor(starts, dtype=torch.float32, device="cpu")[:, None]
    native_pixels = native_pixels + torch.arange(length, dtype=torch.float32, device="cpu")
    lattice_pixels = origins + torch.arange(-halo, extent + halo, dtype=torch.float32, device="cpu")
    return (
        lattice_pixels,
        _resize_axis(native_pixels, lattice, native, align_corners) - origins,
        extent,
    )


def _grid(rows: Tensor, cols: Tensor, source_size: _Size) -> Tensor:
    height, width = source_size
    y = rows.clamp(0, height - 1) * (2 / max(height - 1, 1)) - 1
    x = cols.clamp(0, width - 1) * (2 / max(width - 1, 1)) - 1
    shape = (len(rows), rows.shape[1], cols.shape[1])
    return torch.stack((x[:, None, :].expand(shape), y[:, :, None].expand(shape)), dim=-1)


def _plan_tiles(
    rects: list[_Rect],
    lattice: _Size,
    native: _Size,
    *,
    align_corners: bool,
    halo: int = 0,
) -> _TilePlan:
    xs, ys, widths, heights = zip(*rects, strict=True)
    rows, native_rows, tile_h = _plan_axis(ys, heights[0], lattice[0], native[0], align_corners, halo)
    cols, native_cols, tile_w = _plan_axis(xs, widths[0], lattice[1], native[1], align_corners, halo)
    return _TilePlan(rows, cols, native_rows.clamp(0, tile_h - 1), native_cols.clamp(0, tile_w - 1))


def _blend_taps(taps: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
    batch, channels = taps.shape[:2]
    height, width = rows.shape[1], cols.shape[1]
    taps = taps.reshape(batch, channels, height, 2, width, 2)
    wx = (cols - cols.floor()).to(taps.device, taps.dtype)[:, None, None, None, :]
    wy = (rows - rows.floor()).to(taps.device, taps.dtype)[:, None, :, None]
    horizontal = torch.lerp(taps[..., 0], taps[..., 1], wx)
    return torch.lerp(horizontal[:, :, :, 0], horizontal[:, :, :, 1], wy)


def _tap_positions(coordinates: Tensor, extent: int) -> Tensor:
    lower = coordinates.floor()
    return torch.stack((lower, (lower + 1).clamp(max=extent - 1)), dim=-1).flatten(1)


def _native_samples(features: Tensor, plan: _TilePlan) -> Tensor:
    rows, cols = plan.native_rows, plan.native_cols
    height, width = features.shape[-2:]
    grid = _grid(_tap_positions(rows, height), _tap_positions(cols, width), (height, width))
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    taps = F.grid_sample(
        features.to(dtype), grid.to(features.device, dtype), mode="nearest", align_corners=True
    )
    return _blend_taps(taps, rows, cols).to(features.dtype)


def _source_tile_axis(coordinates: Tensor, size: int) -> tuple[Tensor, Tensor]:
    # Coordinates are monotone and on CPU. Bound both bilinear taps, then
    # shift border tiles inward so all tiles can use the same extent.
    coordinates = coordinates.clamp(0, size - 1)
    starts = coordinates[:, :1].floor().long()
    stops = coordinates[:, -1:].ceil().long() + 1
    extent = int((stops - starts).max())
    starts = starts.clamp(max=size - extent)
    indices = starts + torch.arange(extent, device=starts.device)
    return indices, coordinates - starts


def _resize_tiles(features: Tensor, plan: _TilePlan, lattice_size: _Size, *, align_corners: bool) -> Tensor:
    source_size = features.shape[-2:]
    rows = _resize_axis(plan.rows, source_size[0], lattice_size[0], align_corners)
    cols = _resize_axis(plan.cols, source_size[1], lattice_size[1], align_corners)
    # Local coordinates preserve fractional precision at large native image offsets.
    row_indices, rows = _source_tile_axis(rows, source_size[0])
    col_indices, cols = _source_tile_axis(cols, source_size[1])
    row_indices = row_indices.to(features.device)
    col_indices = col_indices.to(features.device)
    tiles = features[0][:, row_indices[:, :, None], col_indices[:, None, :]]
    channels, count, tile_height, tile_width = tiles.shape
    atlas = tiles.reshape(1, channels, count * tile_height, tile_width)
    offsets = torch.arange(count, device=rows.device)[:, None] * tile_height
    # Fetch integer taps in one packed call. Applying subpixel weights in local
    # coordinates avoids losing low bits when many tiles share a tall atlas.
    grid = _grid(
        _tap_positions(rows, tile_height) + offsets, _tap_positions(cols, tile_width), atlas.shape[-2:]
    )
    height, width = rows.shape[1], cols.shape[1]
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    grid = grid.reshape(1, 2 * count * height, 2 * width, 2).to(features.device, dtype)
    taps = F.grid_sample(atlas.to(dtype), grid, mode="nearest", align_corners=True)
    taps = taps.reshape(channels, count, 2 * height, 2 * width).transpose(0, 1)
    return _blend_taps(taps, rows, cols).to(features.dtype)


@torch.inference_mode()
def fast_depth_regions(head: nn.Module, ctx: DepthContext, rects: Iterable[Sequence[int]]) -> list[Tensor]:
    """Return native Bx1xHxW crops for the Depth Anything V2 model in query order."""
    rectangles = _validated_rects(rects, ctx.native_size)
    tail = head.depth_head.scratch.output_conv2
    conv = tail[0]
    results = {}
    height, width = ctx.decoder_size
    for group in _groups(rectangles):
        plan = _plan_tiles(group, ctx.decoder_size, ctx.native_size, align_corners=True, halo=1)
        features = _resize_tiles(ctx.features, plan, ctx.decoder_size, align_corners=True)

        # The interpolation clamps to the border. The 3x3 sees physical zero padding.
        invalid_rows = (plan.rows < 0) | (plan.rows >= height)
        invalid_cols = (plan.cols < 0) | (plan.cols >= width)
        if invalid_rows.any() or invalid_cols.any():
            invalid = invalid_rows[:, None, :, None] | invalid_cols[:, None, None, :]
            features.masked_fill_(invalid.to(device=features.device), 0)

        depth = F.conv2d(features, conv.weight, conv.bias)
        for i in range(1, len(tail)):
            depth = tail[i](depth)
        crops = _native_samples(F.relu(depth), plan)
        results.update(zip(group, crops.split(1), strict=True))
    return [results[rect] for rect in rectangles]


@torch.inference_mode()
def fast_seg_regions(head: nn.Module, ctx: SegContext, rects: Iterable[Sequence[int]]) -> list[Tensor]:
    """Return native BxCxHxW logits for an eval SegFormer in query order."""
    decoder = head.decode_head
    if decoder.batch_norm.training or decoder.dropout.training:
        raise ValueError("SegFormer decoder batch norm and dropout must be in eval mode")
    rectangles = _validated_rects(rects, ctx.native_size)
    if not rectangles:
        return []

    # Callers can assemble the fold before timing queries.
    folded = getattr(head, "folded", None)
    if folded is None:
        folded = FoldedSegformer(head)
    finest = ctx.hidden_states[0].shape[-2:]
    results = {}
    for group in _groups(rectangles):
        plan = _plan_tiles(group, finest, ctx.native_size, align_corners=False)
        total = None
        for hidden, weight in zip(ctx.hidden_states, folded.kernels, strict=True):
            branch = F.conv2d(_resize_tiles(hidden, plan, finest, align_corners=False), weight)
            total = branch if total is None else total + branch
        fused = decoder.activation(decoder.batch_norm(total + folded.bias))
        logits = decoder.classifier(decoder.dropout(fused))
        crops = _native_samples(logits, plan)
        results.update(zip(group, crops.split(1), strict=True))
    return [results[rect] for rect in rectangles]
