"""Request different native rectangles from released F3 heads on one event packet."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from query_f3.inference.loaders.published_f3 import load_published_f3
from query_f3.inference.loaders.released_flow import load_head
from query_f3.multitask.joint import serve, validate
from query_f3.multitask.models import load_global_head
from query_f3.multitask.service import FeatureService, merged_suffix, packet


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "requests", "output", "upstream", "encoder"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--models", type=Path, help="directory holding verified depth.pth, semantics.pth and segformer.json"
    )
    p.add_argument("--flow-weights", type=Path)
    p.add_argument("--flow-config", type=Path)
    p.add_argument("--size", nargs=2, type=int, required=True, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--method", choices=("exact", "dense", "preview"), default="exact")
    p.add_argument("--split", type=int, choices=range(1, 7), default=3)
    args = p.parse_args()
    torch.set_num_threads(4)
    size = tuple(args.size)
    req = validate(json.loads(args.requests.read_text()), size)
    raw = np.load(args.events, allow_pickle=False)
    if (
        raw.ndim != 2
        or raw.shape[1] not in (3, 4)
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any((raw[:, 0] < -20000) | (raw[:, 0] >= 0))
        or min(size) <= 0
        or np.any((raw[:, 1:3] < 0) | (raw[:, 1:3] >= size))
        or np.any(np.diff(raw[:, 0].astype("i8")) < 0)
    ):
        p.error("expected sorted integer [relative_us,x,y,(polarity)] events in [-20000,0)")
    if args.method == "preview" and not set(req) <= {"features", "occupancy"}:
        p.error("preview supports features and occupancy")
    enc = load_published_f3(args.upstream, args.encoder, args.device).eval()
    enc.return_logits = False
    enc.return_feat = True
    inputs = packet(raw, size)
    heads = {}
    for task in req:
        if task in ("depth", "semantics"):
            if args.models is None:
                p.error("--models is required for global heads")
            heads[task] = load_global_head(
                task,
                args.models / (task + ".pth"),
                args.encoder,
                args.upstream,
                args.models / "segformer.json",
                args.device,
            )
            if task == "semantics" and args.method == "exact":
                from query_f3.multitask.attention import use_sdpa
                from query_f3.multitask.fusion import FoldedSegformer

                heads[task] = use_sdpa(heads[task], task)
                heads[task].folded = FoldedSegformer(heads[task])
        if task == "flow":
            if not args.flow_weights or not args.flow_config:
                p.error("flow requires --flow-weights and --flow-config")
            heads[task] = load_head(
                args.upstream, args.flow_weights, args.flow_config, args.encoder, args.device
            )
    if args.method == "preview":
        service = FeatureService()
        groups = defaultdict(list)
        for r in dict.fromkeys(tuple(r) for boxes in req.values() for r in boxes):
            groups[r[2:]].append(r)
        tiles = {}
        ctx = service.context(enc, inputs.grid(enc), "merged", args.split, 2)
        for boxes in groups.values():
            batch = merged_suffix(enc, ctx, boxes)
            tiles.update(zip(boxes, batch.split(1), strict=True))
        answer = {
            t: [(enc.pred(tiles[r]) if t == "occupancy" else tiles[r]).transpose(2, 3) for r in boxes]
            for t, boxes in req.items()
        }
    else:
        answer = serve(enc, heads, inputs, req, regional=args.method == "exact")
    arrays = {
        f"{t}_{i}": v[0].float().cpu().numpy() for t, tiles in answer.items() for i, v in enumerate(tiles)
    }
    arrays.update({t + "_rectangles": np.asarray(boxes, dtype="i4") for t, boxes in req.items()})
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
