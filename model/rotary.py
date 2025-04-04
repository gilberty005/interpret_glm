import torch
from typing import Tuple
from einops import repeat
from torch import autocast


def rotate_half(x):
    "from https://github.com/facebookresearch/esm"
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


#@torch.jit.script   # would require setting shape to static (or finite number of shapes)
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int = -2, fixed_len=False):
    "from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py"
    # NOTE: This could probably be moved to Triton
    # Handle a possible sequence length mismatch in between q and k
    # TODO(eric): Allow fixed_len to be set in config. Only standard rope requires it
    if len(cos.shape) == 2: # cos, sin is unbatched. (non-fragment approach)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if not fixed_len:
        cos = cos[:, :x.shape[seq_dimension], :]
        sin = sin[:, :x.shape[seq_dimension], :]
    if seq_dimension == -3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    cos = cos.to(x.device)
    sin = sin.to(x.device)
    return (x.to(cos.dtype) * cos) + (rotate_half(x.to(cos.dtype)) * sin)
