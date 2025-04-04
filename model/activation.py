import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwisGLU activation , combine with fc1
    Replaces the first linear layer of FFN.
    SWISH: A SELF-GATED ACTIVATION FUNCTION arXiv:1710.05941
    GLU Variants Improve Transformer  arXiv:2002.05202v1
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out*2, bias=bias)
    
    def forward(self, x):
        x = self.linear(x)
        return F.silu(x[..., :self.dim_out]) * x[..., self.dim_out:]  # gate * x


class SwiGLUShard(nn.Module):
    'Unfused swiglu to support sharding'
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.linear2 = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.silu(self.linear1(x)) * self.linear2(x)  # gate * x


class GeGLU(nn.Module):
    """GeGLU activation , combine with fc1
    Replaces the first linear layer of FFN.
    SWISH: A SELF-GATED ACTIVATION FUNCTION arXiv:1710.05941
    GLU Variants Improve Transformer  arXiv:2002.05202v1
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out*2, bias=bias)
    
    def forward(self, x):
        x = self.linear(x)
        return F.gelu(x[..., :self.dim_out]) * x[..., self.dim_out:]  # gate * x
