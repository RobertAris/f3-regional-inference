"""Regional tails for the released F3 depth and semantic segmentation heads.

Call the context functions once per full native NCHW field with the corresponding
model in eval mode. DINOv2 and the DPT refinement prefix, and all four SegFormer
encoder stages, still run on the whole image. Subsequent queries only evaluate
the decoder tails on the support of the requested native rectangles.
"""

from collections.abc import Iterable, Sequence
from operator import index
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_Size = tuple[int, int]
_Rect = tuple[int, int, int, int]


class DepthContext(NamedTuple):
    features: Tensor
    native_size: _Size
    decoder_size: _Size


class SegContext(NamedTuple):
    hidden_states: tuple[Tensor, ...]
    native_size: _Size


class _AxisSamples(NamedTuple):
    lower: list[int]
    upper: list[int]
    fraction: list[float]


def _native_size(field: Tensor) -> _Size:
    if field.ndim != 4 or field.shape[0] != 1 or min(field.shape[-2:]) < 1:
        raise ValueError("expected a nonempty B=1 NCHW field")
    return tuple(field.shape[-2:])


def _validated_rects(rects: Iterable[Sequence[int]], native_size: _Size) -> list[_Rect]:
    height, width = native_size
    result = []
    for rect in rects:
        try:
            x, y, w, h = map(index, rect)
        except (TypeError, ValueError) as error:
            raise ValueError("each rectangle must contain four integers (x, y, w, h)") from error
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(f"rectangle {(x, y, w, h)} is outside the native image {width}x{height}")
        result.append((x, y, w, h))
    return result


def _axis_samples(
    positions: Sequence[int], source_size: int, target_size: int, align_corners: bool
) -> _AxisSamples:
    # Represent each source coordinate as a rational to determine lattice
    # support without rounding at integral coordinates or physical borders.
    if align_corners:
        scale, offset, denominator = source_size - 1, 0, max(target_size - 1, 1)
    else:
        scale, offset, denominator = 2 * source_size, source_size - target_size, 2 * target_size
    last = (source_size - 1) * denominator
    lower, upper, fractions = [], [], []
    for position in positions:
        coordinate = min(max(scale * position + offset, 0), last)
        pixel, remainder = divmod(coordinate, denominator)
        lower.append(pixel)
        upper.append(pixel + (remainder != 0))
        fractions.append(remainder / denominator)
    return _AxisSamples(lower, upper, fractions)


def _axis_support(samples: _AxisSamples) -> tuple[list[int], _AxisSamples]:
    positions = sorted(set(samples.lower).union(samples.upper))
    local = {position: i for i, position in enumerate(positions)}
    return positions, _AxisSamples(
        [local[i] for i in samples.lower],
        [local[i] for i in samples.upper],
        samples.fraction,
    )


def _crop_support(
    rect: _Rect, source_size: _Size, native_size: _Size, align_corners: bool
) -> tuple[list[int], list[int], _AxisSamples, _AxisSamples]:
    x, y, w, h = rect
    source_h, source_w = source_size
    native_h, native_w = native_size
    rows, row_samples = _axis_support(_axis_samples(range(y, y + h), source_h, native_h, align_corners))
    cols, col_samples = _axis_support(_axis_samples(range(x, x + w), source_w, native_w, align_corners))
    return rows, cols, row_samples, col_samples


def _bilinear(features: Tensor, rows: _AxisSamples, cols: _AxisSamples) -> Tensor:
    device = features.device
    y0 = torch.tensor(rows.lower, device=device)[:, None]
    x0 = torch.tensor(cols.lower, device=device)
    if rows.lower == rows.upper and cols.lower == cols.upper:
        return features[..., y0, x0]

    y1 = torch.tensor(rows.upper, device=device)[:, None]
    x1 = torch.tensor(cols.upper, device=device)
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    wy = torch.tensor(rows.fraction, device=device, dtype=dtype)[:, None]
    wx = torch.tensor(cols.fraction, device=device, dtype=dtype)
    top = (1 - wx) * features[..., y0, x0] + wx * features[..., y0, x1]
    bottom = (1 - wx) * features[..., y1, x0] + wx * features[..., y1, x1]
    return ((1 - wy) * top + wy * bottom).to(features.dtype)


def _resize_at(
    features: Tensor, rows: Sequence[int], cols: Sequence[int], target_size: _Size, align_corners: bool
) -> Tensor:
    return _bilinear(
        features,
        _axis_samples(rows, features.shape[-2], target_size[0], align_corners),
        _axis_samples(cols, features.shape[-1], target_size[1], align_corners),
    )


@torch.inference_mode()
def depth_context(dav2: nn.Module, field_yx: Tensor) -> DepthContext:
    """Run the full-image ViT-B and DPT prefix through `output_conv1`."""
    h, w = native_size = _native_size(field_yx)
    decoder_width = (int(w / h * 518) + 13) // 14 * 14
    if decoder_width == 0:
        raise ValueError("the F3 depth aspect transform yields a zero-width image")
    decoder_size = (518, decoder_width)
    patch_h, patch_w = (length // 14 for length in decoder_size)
    image = F.interpolate(field_yx, size=decoder_size, mode="bilinear", align_corners=False)
    tokens = dav2.pretrained.get_intermediate_layers(
        image, dav2.intermediate_layer_idx[dav2.encoder], return_class_token=True
    )

    head = dav2.depth_head
    layers = []
    for (patch_tokens, _), project, resize in zip(tokens, head.projects, head.resize_layers, strict=True):
        patch_map = patch_tokens.transpose(1, 2).reshape(1, patch_tokens.shape[-1], patch_h, patch_w)
        layers.append(resize(project(patch_map)))

    scratch = head.scratch
    layer1 = scratch.layer1_rn(layers[0])
    layer2 = scratch.layer2_rn(layers[1])
    layer3 = scratch.layer3_rn(layers[2])
    layer4 = scratch.layer4_rn(layers[3])
    path4 = scratch.refinenet4(layer4, size=layer3.shape[-2:])
    path3 = scratch.refinenet3(path4, layer3, size=layer2.shape[-2:])
    path2 = scratch.refinenet2(path3, layer2, size=layer1.shape[-2:])
    path1 = scratch.refinenet1(path2, layer1)
    return DepthContext(scratch.output_conv1(path1), native_size, decoder_size)


def _conv_axis(positions: list[int]) -> tuple[Sequence[int], int]:
    if positions[-1] - positions[0] + 1 == len(positions):
        return range(positions[0] - 1, positions[-1] + 2), 1

    # Downsampling can skip decoder sites. Pack the required three-tap
    # neighborhoods so stride three applies the same kernel across the gaps.
    return [neighbor for pixel in positions for neighbor in (pixel - 1, pixel, pixel + 1)], 3


def _depth_tail(tail: nn.Sequential, context: DepthContext, rows: list[int], cols: list[int]) -> Tensor:
    halo_y, stride_y = _conv_axis(rows)
    halo_x, stride_x = _conv_axis(cols)
    height, width = context.decoder_size
    features = _resize_at(context.features, halo_y, halo_x, context.decoder_size, align_corners=True)

    # A halo position outside the resized image is physical convolution
    # padding. Interpolation itself uses border values even on partial tiles.
    if rows[0] == 0 or rows[-1] == height - 1:
        invalid = torch.tensor([y < 0 or y >= height for y in halo_y], device=features.device)
        features.masked_fill_(invalid[:, None], 0)
    if cols[0] == 0 or cols[-1] == width - 1:
        invalid = torch.tensor([x < 0 or x >= width for x in halo_x], device=features.device)
        features.masked_fill_(invalid, 0)

    conv = tail[0]
    depth = F.conv2d(features, conv.weight, conv.bias, stride=(stride_y, stride_x))
    for i in range(1, len(tail)):
        depth = tail[i](depth)
    return F.relu(depth)


@torch.inference_mode()
def depth_regions(dav2: nn.Module, context: DepthContext, rects: Iterable[Sequence[int]]) -> list[Tensor]:
    """Return Bx1xHxW depth crops in native query order."""
    tail = dav2.depth_head.scratch.output_conv2
    crops = {}
    rectangles = _validated_rects(rects, context.native_size)
    for rect in dict.fromkeys(rectangles):
        rows, cols, row_samples, col_samples = _crop_support(
            rect, context.decoder_size, context.native_size, align_corners=True
        )
        depth = _depth_tail(tail, context, rows, cols)
        crops[rect] = _bilinear(depth, row_samples, col_samples)
    return [crops[rect] for rect in rectangles]


@torch.inference_mode()
def seg_context(segformer: nn.Module, field_yx: Tensor) -> SegContext:
    """Run the full-image HuggingFace SegFormer encoder and retain its four NCHW stages."""
    native_size = _native_size(field_yx)
    outputs = segformer.segformer(field_yx, output_hidden_states=True, return_dict=True)
    return SegContext(outputs.hidden_states, native_size)


@torch.inference_mode()
def seg_regions(segformer: nn.Module, context: SegContext, rects: Iterable[Sequence[int]]) -> list[Tensor]:
    """Return BxCxHxW semantic logits in native query order."""
    head = segformer.decode_head
    if head.batch_norm.training or head.dropout.training:
        raise ValueError("SegFormer decoder batch norm and dropout must be in eval mode")

    finest_size = context.hidden_states[0].shape[-2:]
    crops = {}
    rectangles = _validated_rects(rects, context.native_size)
    for rect in dict.fromkeys(rectangles):
        rows, cols, row_samples, col_samples = _crop_support(
            rect, finest_size, context.native_size, align_corners=False
        )
        projected = []
        for hidden, mlp in zip(context.hidden_states, head.linear_c, strict=True):
            # An affine channel projection commutes with border-clamped
            # bilinear interpolation. The nonlinear fusion follows sampling.
            tile = _resize_at(hidden, rows, cols, finest_size, align_corners=False)
            projected.append(mlp.proj(tile.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        fused = head.linear_fuse(torch.cat(projected[::-1], dim=1))
        fused = head.dropout(head.activation(head.batch_norm(fused)))
        crops[rect] = _bilinear(head.classifier(fused), row_samples, col_samples)

    return [crops[rect] for rect in rectangles]
