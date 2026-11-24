"""Frozen supplementary readout from the authors' released F3 flow model."""

import ast
import hashlib
import math
from collections import OrderedDict
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_FLOW_CONFIG_SHA256 = "8a6332396d8f157a3da029eb272945306eff8786e71668c822c66d673f2611eb"
_FLOW_CHECKPOINT_SHA256 = "af1a08a637919d19c42af43a609e3c2ecb05fdb3b6009be0d40afd9d7aa63ace"
# Official encoder digest recorded by diagnostics.published_f3.CHECKPOINT_SHA256.
_PUBLISHED_CHECKPOINT_SHA256 = "deb09ebe626365ae0c960ae5a96755f37f89d17a245d86d2f7168783a94d8baf"
_COMPILE_PREFIX = "_orig_mod."
_ENCODER_PREFIX = "eventff." + _COMPILE_PREFIX
_HEAD_PREFIX = "flowhead." + _COMPILE_PREFIX
_CHANNELS = 32
_RATE_INTERVAL_US = 100_000


def _read_flow_config(path: str | Path) -> dict[str, list[int]]:
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != _FLOW_CONFIG_SHA256:
        raise ValueError(f"Flow config SHA-256 mismatch for {path}")

    # The verified YAML stores each flowhead field as an inline integer list.
    _, _, fields = content.decode("utf-8").partition("\nflowhead:\n")
    config = {}
    for line in fields.splitlines():
        if not line.startswith("  "):
            break
        key, value = line.strip().split(": ", 1)
        config[key] = ast.literal_eval(value)
    return config


def _load_verified_state(path: str | Path, expected_digest: str) -> dict:
    with Path(path).open("rb") as checkpoint:
        if hashlib.file_digest(checkpoint, "sha256").hexdigest() != expected_digest:
            raise ValueError(f"Checkpoint SHA-256 mismatch for {path}")
        checkpoint.seek(0)
        return torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]


def _load_hub_block(f3_repo: str | Path) -> type[nn.Module]:
    # Load the Block and its torch layers from the trusted, pinned local checkout.
    path = Path(f3_repo) / "hubconf.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"LayerNorm", "GRN", "Block"}
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if len(classes) != len(names) or {node.name for node in classes} != names:
        raise ValueError(f"Expected the pinned F3 Block and its layers in {path}")
    module = ast.Module(body=classes, type_ignores=[])
    namespace = {"__name__": __name__, "torch": torch, "nn": nn, "F": F}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Block"]


def load_head(
    f3_repo: str | Path,
    flow_checkpoint: str | Path,
    flow_config: str | Path,
    published_checkpoint: str | Path,
    device: str | torch.device,
) -> nn.Module:
    """Verify the release artifacts and restore the frozen full-frame head."""
    config = _read_flow_config(flow_config)
    bundled = _load_verified_state(flow_checkpoint, _FLOW_CHECKPOINT_SHA256)
    published = _load_verified_state(published_checkpoint, _PUBLISHED_CHECKPOINT_SHA256)

    bundled_encoder = {}
    head_state = {}
    for key, tensor in bundled.items():
        if key.startswith(_ENCODER_PREFIX):
            bundled_encoder[key.removeprefix("eventff.")] = tensor
        elif key.startswith(_HEAD_PREFIX):
            head_state[key.removeprefix(_HEAD_PREFIX)] = tensor
        else:
            raise ValueError(f"Unexpected bundled flow checkpoint key {key!r}")

    if any(not key.startswith(_COMPILE_PREFIX) for key in published):
        raise ValueError("Expected compiled published encoder keys")
    if not published or bundled_encoder.keys() != published.keys():
        raise ValueError("Bundled encoder keys differ from the published encoder keys")
    for key, reference in published.items():
        tensor = bundled_encoder[key]
        if tensor.dtype != reference.dtype or not torch.equal(tensor, reference):
            raise ValueError(f"Bundled encoder tensor differs from the published tensor at {key!r}")

    block = _load_hub_block(f3_repo)
    decoder = nn.Sequential(*(
        block(_CHANNELS, kernel_size=kernel, bottleneck=bottleneck, dilation=dilation)
        for kernel, bottleneck, dilation in zip(
            config["kernels"], config["btlncks"], config["dilations"], strict=True
        )
    ))
    head = nn.Sequential(OrderedDict([
        ("decoder", decoder),
        ("pred", nn.Conv2d(_CHANNELS, 2, kernel_size=1)),
    ]))
    head.load_state_dict(head_state, strict=True, assign=True)
    return head.to(device).requires_grad_(False).eval()


@torch.inference_mode()
def sample_flow(
    head: nn.Module, field: torch.Tensor, pixels, interval_us: int | float
) -> torch.Tensor:
    """Gather native (x, y) displacements from a batch-one, width-first field."""
    assert isinstance(field, torch.Tensor) and field.ndim == 4, "Expected a [1,W,H,32] field"
    batch, width, height, channels = field.shape
    assert batch == 1 and width > 0 and height > 0 and channels == _CHANNELS, (
        "Expected a nonempty [1,W,H,32] field"
    )
    assert field.is_floating_point(), "Expected a floating-point feature field"
    assert isinstance(interval_us, Real) and not isinstance(interval_us, bool), (
        "Expected a numeric flow interval in microseconds"
    )
    assert math.isfinite(interval_us) and interval_us > 0, "Expected a positive finite flow interval"

    try:
        indices = torch.as_tensor(pixels)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise AssertionError("Expected native integer pixels with shape [P,2]") from error
    assert indices.ndim == 2 and indices.shape[1] == 2, "Expected pixels with shape [P,2]"
    assert indices.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.uint16, torch.uint32, torch.uint64,
    ), "Expected integer pixel coordinates"
    indices = indices.to(dtype=torch.long)
    x, y = indices.unbind(dim=1)
    outside = (x < 0) | (x >= width) | (y < 0) | (y >= height)
    assert not outside.any().item(), "Native pixel coordinates are outside the feature field"
    x, y = indices.to(device=field.device).unbind(dim=1)

    # The Hub Block normalizes over spatial dimensions, so evaluate the whole frame.
    flow = head(field.permute(0, 3, 2, 1))
    assert flow.shape == (1, 2, height, width), "Expected a [1,2,H,W] flow field"
    displacement = flow[0, :, y, x].T * (interval_us / _RATE_INTERVAL_US)
    assert torch.isfinite(displacement).all().item(), "Sampled flow displacement is non-finite"
    return displacement
