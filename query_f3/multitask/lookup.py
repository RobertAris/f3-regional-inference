"""Resolution-specific cache of the released encoder's discrete event embeddings."""

import numpy as np
import torch

from query_f3.engine import Packet


class LookupPacket(Packet):
    def __init__(self, events, size, table):
        if tuple(table.shape[:3]) != (*size, 20):
            raise ValueError("Embedding cache was built for a different sensor")
        self.size = size
        self.bucket = None
        self.table = table
        self.xy = events[:, 1:3]
        self.age = (-1 - events[:, 0].astype("int64")) // 1000
        if (self.age < 0).any() or (self.age >= 20).any():
            raise ValueError("Event outside 20 ms packet")
        Packet.__post_init__(self)

    @property
    def ev(self):
        # Only the unmodified released reference needs floating-point coordinates.
        return np.column_stack((self.xy / np.asarray(self.size), self.age / 20)).astype("f")

    def grid(self, enc, rect=None):
        table = self.table
        x, y, w, h = (0, 0, *self.size) if rect is None else rect
        take = slice(None) if rect is None else self.subset(rect)
        pos = torch.as_tensor(np.column_stack((self.xy[take], self.age[take])), device=table.device)
        v = table[pos[:, 0], pos[:, 1], pos[:, 2]]
        output = table.new_zeros((w, h, table.shape[-1]))
        output.index_put_((pos[:, 0] - x, pos[:, 1] - y), v, accumulate=True)
        return output.permute(2, 0, 1)[None]


@torch.inference_mode()
def build_lookup(enc, size):
    """Generate exactly the same table as the upstream fixed-resolution cache in chunks."""
    w, h = size
    dev = next(enc.parameters()).device
    c = enc.downsample_layers[0][0].in_channels
    table = torch.empty((w * h * 20, c), device=dev, dtype=torch.float32)
    for a in range(0, len(table), 100000):
        stop = min(a + 100000, len(table))
        i = torch.arange(a, stop, device=dev)
        coords = torch.stack(
            ((i // (h * 20)).float() / w, ((i // 20) % h).float() / h, (i % 20).float() / 20), -1
        )
        table[a:stop] = enc.multi_hash_encoder(coords[None])[0]
    return table.reshape(w, h, 20, c)
