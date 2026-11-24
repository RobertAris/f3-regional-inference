"""Load the two released global task heads without CUDA-specific trainer code."""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from query_f3.binding import digest
from query_f3.inference.loaders.published_f3 import CHECKPOINT_SHA256

HASHES = {
    "depth": "26055d44352768920c772501f1950d3fe1eccef7a80c373b9e12df7df115fad3",
    "semantics": "797dea04860bfdfa3c95357796ca38e279e555ce758914df4c8103dbea974712",
    "segformer.json": "83a272be23218b57a9cc7f5b034f4aa888362b5809097b8321a4154e650b1cd5",
}


def load_global_head(task, weights, encoder_weights, f3_repo, seg_config=None, device="cpu"):
    """Verify the release bytes and bundled encoder before sharing one field."""
    if task not in ("depth", "semantics") or digest(weights) != HASHES[task]:
        raise ValueError(f"Unknown task or checkpoint digest for {task}")
    if digest(encoder_weights) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected standalone F3 checkpoint")
    safe = [np._core.multiarray.scalar, np.dtype, type(np.dtype("f8")), type(np.dtype("f4"))]
    with torch.serialization.safe_globals(safe):
        state = torch.load(weights, weights_only=True, mmap=True, map_location="cpu")["model"]
    reference = torch.load(encoder_weights, weights_only=True, mmap=True, map_location="cpu")["model"]
    bundled = {k.removeprefix("eventff."): v for k, v in state.items() if k.startswith("eventff.")}
    if reference.keys() != bundled.keys() or any(
        a.dtype != bundled[k].dtype or not torch.equal(a, bundled[k]) for k, a in reference.items()
    ):
        raise ValueError("Bundled and standalone F3 encoder tensors differ")
    if task == "depth":
        source = str(Path(f3_repo).resolve() / "src/f3/tasks/depth")
        sys.path.insert(0, source)
        try:
            cls = importlib.import_module("depth_anything_v2.dpt").DepthAnythingV2
        finally:
            sys.path.remove(source)
        model = cls(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
        model.pretrained.patch_embed.proj = torch.nn.Conv2d(32, 768, 14, stride=14)
        prefix = "dav2."
    else:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if seg_config is None or digest(seg_config) != HASHES["segformer.json"]:
            raise ValueError("Expected the released SegFormer B3 architecture config")
        config = SegformerConfig.from_json_file(str(seg_config))
        config.num_labels = 11
        config.num_channels = 32
        model = SegformerForSemanticSegmentation(config)
        prefix = "segformer."
    head = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    if len(head) + len(bundled) != len(state):
        raise ValueError("Unexpected parameters in released checkpoint")
    model.load_state_dict(head, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device)


def dense_output(task, head, field_yx, *, depth_height=518):
    """Official dense, native-output forward for a B,C,H,W encoder field."""
    h, w = field_yx.shape[-2:]
    if task == "depth":
        width = max(14, 14 * math.ceil(int(w / h * depth_height) / 14))
        features = F.interpolate(field_yx, (depth_height, width), mode="bilinear", align_corners=False)
        logits = head(features)[:, None]
        return F.interpolate(logits, (h, w), mode="bilinear", align_corners=True)
    if task == "semantics":
        logits = head(field_yx).logits
        return F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
    raise ValueError(task)
