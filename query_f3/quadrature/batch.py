"""Batched native regional inference with fixed suffix factors."""

from operator import index

import torch

from ..engine import network
from ..norm_block import apply_block


@torch.no_grad()
def batched_suffix(enc, head, ctx, rects):
    """Return width-first encoder fields and optional flow for equal-size ROIs.

    Each rectangle is `(x, y, width, height)` in native pixels. The field has
    shape `[Q, C, width, height]`. With `head=None`, `ctx` supplies six fixed
    encoder factors and a prefix depth from 1 to 6, and flow is `None`.
    With a head, `ctx` supplies ten factors and a depth from 3 to 6, and
    unscaled flow has shape `[Q, 2, width, height]`.
    """
    field_only = head is None
    min_depth = 1 if field_only else 3
    depth_error = f"prefix depth must be an integer from {min_depth} to 6"
    try:
        depth = index(ctx.depth)
    except TypeError as exc:
        raise ValueError(depth_error) from exc
    if not min_depth <= depth <= 6:
        raise ValueError(depth_error)

    state = ctx.state
    if (
        not isinstance(state, torch.Tensor)
        or state.ndim != 4
        or state.shape[0] != 1
        or min(state.shape[1:]) <= 0
    ):
        raise ValueError("prefix state must have shape [1, C, W, H] with positive C, W, H")
    gain_count, gain_name = (6, "six") if field_only else (10, "ten")
    gain_error = f"{gain_name} fixed channel factor vectors are required"
    try:
        gains = tuple(ctx.gains)
    except TypeError as exc:
        raise ValueError(gain_error) from exc
    if len(gains) != gain_count or any(
        not isinstance(gain, torch.Tensor) or gain.ndim != 1 or gain.numel() == 0
        for gain in gains
    ):
        raise ValueError(gain_error)

    try:
        rects = [tuple(map(index, rect)) for rect in rects]
    except TypeError as exc:
        raise ValueError("each ROI must be (x, y, width, height) in integer pixels") from exc
    if not rects or any(len(rect) != 4 for rect in rects):
        raise ValueError("one or more (x, y, width, height) ROIs are required")

    frame_width, frame_height = state.shape[-2:]
    _, _, width, height = rects[0]
    if width <= 0 or height <= 0 or any(
        (w, h) != (width, height)
        or x < 0
        or y < 0
        or x + w > frame_width
        or y + h > frame_height
        for x, y, w, h in rects
    ):
        raise ValueError("ROIs must be contained in the frame and share a positive extent")

    # The native suffix has 3 pixels of radius per remaining encoder block
    # and 4 per flow block. Gather the whole halo for each ROI in one tensor.
    margin = 3 * (6 - depth) + (0 if field_only else 16)
    coords = torch.tensor(rects, dtype=torch.long, device=state.device)
    xs = coords[:, 0, None] + torch.arange(-margin, width + margin, device=state.device)
    ys = coords[:, 1, None] + torch.arange(-margin, height + margin, device=state.device)
    outside_x = (xs < 0) | (xs >= frame_width)
    outside_y = (ys < 0) | (ys >= frame_height)
    outside = outside_x[:, None, :, None] | outside_y[:, None, None, :]
    xs = xs.clamp(0, frame_width - 1)
    ys = ys.clamp(0, frame_height - 1)
    pixels = state[0].permute(1, 2, 0)
    values = pixels[xs[:, :, None], ys[:, None, :]].permute(0, 3, 1, 2)

    block_index = 0
    field = None
    for stem, blocks, transpose in network(enc, head):
        if transpose:
            field = values[..., margin : margin + width, margin : margin + height]
            values = values.transpose(-2, -1)
            outside = outside.transpose(-2, -1)
        elif block_index >= depth:
            values = stem(values)

        for block in blocks:
            if block_index >= depth:
                # Stems and block biases can produce nonzero values here.
                # Reapply the physical padding before the next convolution.
                values, _ = apply_block(
                    block, values.masked_fill(outside, 0), fixed=gains[block_index]
                )
                radius = block.dwconv.kernel_size[0] // 2
                crop_x = slice(radius, values.shape[-2] - radius)
                crop_y = slice(radius, values.shape[-1] - radius)
                values = values[..., crop_x, crop_y]
                outside = outside[..., crop_x, crop_y]
                margin -= radius
            block_index += 1

    if field_only:
        return values, None
    return field, head.pred(values).transpose(-2, -1)
