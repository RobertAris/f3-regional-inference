"""One field per packet for multiple native output requests."""

from operator import index

import torch

from query_f3.multitask.decoders import depth_context, depth_regions, seg_context, seg_regions
from query_f3.multitask.models import dense_output
from query_f3.multitask.service import encoder_field, gather_yx

TASKS = ("depth", "semantics", "flow", "features", "occupancy")


def validate(requests, size):
    if not requests or any(key not in TASKS for key in requests):
        raise ValueError(f"Requests must use task names from {TASKS}")
    result = {}
    for task, specs in requests.items():
        rects = [tuple(map(index, r)) for r in specs]
        if not rects or any(len(r) != 4 for r in rects):
            raise ValueError(f"Expected rectangles for {task}")
        if any(
            min(x, y) < 0 or min(w, h) < 1 or x + w > size[0] or y + h > size[1] for x, y, w, h in rects
        ):
            raise ValueError(f"Rectangle outside sensor for {task}")
        result[task] = rects
    return result


@torch.inference_mode()
def prepare(enc, heads, p, requests, *, regional, batched=False):
    """Build same-packet context for an exact request. B,C,H,W throughout this API."""
    requests = validate(requests, p.size)
    field = encoder_field(enc, p.grid(enc)).transpose(2, 3)
    context = {}
    for task in requests:
        if task in ("depth", "semantics"):
            head = heads[task]
            if regional:
                context[task] = (depth_context if task == "depth" else seg_context)(head, field)
            elif task == "semantics" and hasattr(head, "folded"):
                states = head.segformer(field, output_hidden_states=True).hidden_states
                context[task] = head.folded(states, field.shape[-2:])
            else:
                context[task] = dense_output(task, head, field)
        elif task == "features":
            context[task] = field
        elif task == "occupancy":
            context[task] = field if regional else enc.pred(field.transpose(2, 3)).transpose(2, 3)
        elif task == "flow":
            # Each released flow GRN uses the entire native spatial plane.
            context[task] = heads[task](field)
    return context


@torch.inference_mode()
def query(enc, heads, context, requests, *, regional, batched=False):
    answer = {}
    for task, rects in requests.items():
        if regional and task in ("depth", "semantics"):
            fn = depth_regions if task == "depth" else seg_regions
            if batched:
                from .batched_decoders import fast_depth_regions, fast_seg_regions

                fn = fast_depth_regions if task == "depth" else fast_seg_regions
            answer[task] = fn(heads[task], context[task], rects)
        else:
            tiles = gather_yx(context[task], rects)
            if regional and task == "occupancy":
                tiles = [enc.pred(v.transpose(2, 3)).transpose(2, 3) for v in tiles]
            answer[task] = tiles
    return answer


@torch.inference_mode()
def serve(enc, heads, p, requests, *, regional=True, batched=True):
    requests = validate(requests, p.size)
    ctx = prepare(enc, heads, p, requests, regional=regional, batched=batched)
    return query(enc, heads, ctx, requests, regional=regional, batched=batched)
