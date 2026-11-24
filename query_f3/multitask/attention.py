"""Equivalent inference attention using PyTorch's scaled-dot-product operator."""

from types import MethodType
from torch.nn import functional as F


def _dino(module, x, attn_bias=None):
    if attn_bias is not None or module.training:
        raise ValueError("Only unmasked evaluation attention is supported")
    b, n, c = x.shape
    q, k, v = (
        module.qkv(x)
        .reshape(b, n, 3, module.num_heads, c // module.num_heads)
        .permute(2, 0, 3, 1, 4)
        .unbind(0)
    )
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
    return module.proj_drop(module.proj(out))


def _segformer(module, x, height, width, output_attentions=False):
    if output_attentions or module.training:
        raise ValueError("Only evaluation without attention-map output is supported")
    b, n, c = x.shape

    def heads(a):
        return a.reshape(b, -1, module.num_attention_heads, module.attention_head_size).transpose(1, 2)

    q = heads(module.query(x))
    if module.sr_ratio > 1:
        x = module.sr(x.transpose(1, 2).reshape(b, c, height, width))
        x = module.layer_norm(x.flatten(2).transpose(1, 2))
    out = F.scaled_dot_product_attention(q, heads(module.key(x)), heads(module.value(x)))
    return (out.transpose(1, 2).reshape(b, n, c),)


def use_sdpa(model, task):
    """Use the same inference backend in dense and regional transformer paths."""
    if task == "depth":
        layers = [b.attn for b in model.pretrained.blocks]
        fn = _dino
    elif task == "semantics":
        layers = [block.attention.self for stage in model.segformer.encoder.block for block in stage]
        fn = _segformer
    else:
        raise ValueError(task)
    for layer in layers:
        layer.forward = MethodType(fn, layer)
    return model
