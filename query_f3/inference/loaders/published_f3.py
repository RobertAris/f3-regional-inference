"""Load the published F3 weights locally and smoke-test a synthetic event batch."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import torch
import yaml


CHECKPOINT_SHA256 = "deb09ebe626365ae0c960ae5a96755f37f89d17a245d86d2f7168783a94d8baf"
MODEL_CONFIG = "confs/ff/modeloptions/1280x720x20_patchff_ds1_small.yml"
COMPILE_PREFIX = "_orig_mod."


def load_published_f3(
    f3_repo: str | Path, checkpoint: str | Path, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Restore the published checkpoint using a trusted local F3 checkout."""
    with Path(checkpoint).open("rb") as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {checkpoint}: "
                f"expected {CHECKPOINT_SHA256}, got {digest}"
            )
        checkpoint_file.seek(0)
        wrapped_state = torch.load(
            checkpoint_file, weights_only=True, map_location="cpu"
        )["model"]

    if not all(key.startswith(COMPILE_PREFIX) for key in wrapped_state):
        raise ValueError(f"Expected every checkpoint key to start with {COMPILE_PREFIX!r}")
    state = {
        key.removeprefix(COMPILE_PREFIX): value for key, value in wrapped_state.items()
    }

    repo = Path(f3_repo)
    config_path = repo / MODEL_CONFIG
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or config.pop("model", None) != "EventPatchFF":
        raise ValueError(f"Expected EventPatchFF in {config_path}")

    event_patch_ff = runpy.run_path(str(repo / "hubconf.py"))["EventPatchFF"]
    # The trainer derives integer resolutions on CPU, where rounding differs from MPS.
    config.update(device="cpu", return_logits=True, return_feat=True)
    model = event_patch_ff(**config)
    hash_encoder = model.multi_hash_encoder
    resolutions = hash_encoder.resolutions
    del hash_encoder.resolutions
    # A non-persistent buffer moves with the model and preserves the checkpoint schema.
    hash_encoder.register_buffer("resolutions", resolutions, persistent=False)
    model.device = device
    model.to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(2)
    model = load_published_f3(args.f3_repo, args.checkpoint, device=args.device)

    with torch.inference_mode():
        events = torch.tensor(
            [[0.25, 0.25, 0.125], [0.5, 0.5, 0.5], [0.75, 0.75, 0.875]],
            dtype=torch.float32,
            device=args.device,
        )
        counts = torch.tensor([3], dtype=torch.int64, device=args.device)
        logits, features = model(events, counts)
        if features.shape != (1, 1280, 720, 32) or logits.shape != (1, 1280, 720, 20):
            raise RuntimeError(
                f"Unexpected F3 shapes. Features: {tuple(features.shape)}, "
                f"logits: {tuple(logits.shape)}"
            )
        all_finite = all(
            torch.isfinite(output).all().item() for output in (features, logits)
        )
        if not all_finite:
            raise RuntimeError("F3 smoke test produced non-finite features or logits")

    report = {
        "device": args.device,
        "hash_grid": model.multi_hash_encoder.resolutions.tolist(),
        "features_shape": list(features.shape),
        "logits_shape": list(logits.shape),
        "all_finite": all_finite,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
