"""Fold the two linear projections in each SegFormer decoder branch for inference."""

import torch
from torch.nn import functional as F


class FoldedSegformer:
    def __init__(self, model):
        head = model.decode_head
        if model.training or head.linear_fuse.bias is not None:
            raise ValueError("Expected the released evaluation decoder")
        # DecodeHead concatenates the coarsest branch first.
        fusion = head.linear_fuse.weight.detach()[:, :, 0, 0].cpu().double()
        parts = fusion.chunk(len(head.linear_c), dim=1)[::-1]
        self.kernels = []
        bias = torch.zeros(fusion.shape[0], dtype=torch.float64)
        for branch, mix in zip(head.linear_c, parts, strict=True):
            proj = branch.proj
            linear = mix @ proj.weight.detach().cpu().double()
            self.kernels.append(linear.to(proj.weight.dtype).to(proj.weight.device)[:, :, None, None])
            bias += mix @ proj.bias.detach().cpu().double()
        self.bias = bias.to(head.linear_fuse.weight.dtype).to(head.linear_fuse.weight.device)[
            None, :, None, None
        ]
        self.head = head

    def __call__(self, states, native_size):
        size = states[0].shape[-2:]
        total = None
        for state, weight in zip(states, self.kernels, strict=True):
            branch = F.conv2d(state, weight)
            if branch.shape[-2:] != size:
                branch = F.interpolate(branch, size=size, mode="bilinear", align_corners=False)
            total = branch if total is None else total + branch
        out = self.head.activation(self.head.batch_norm(total + self.bias))
        logits = self.head.classifier(self.head.dropout(out))
        return F.interpolate(logits, size=native_size, mode="bilinear", align_corners=False)
