import torch.nn as nn
from mup import MuReadout
import math


class MuReadoutWrap(MuReadout):
    '''Removes inplace op allowing compile'''
    def __init__(self, *args, **kwargs):
        self.static_width = None
        super().__init__(*args, **kwargs)
        
    def width_mult(self):
        assert hasattr(self.weight, 'infshape'), (
            'Please call set_base_shapes(...). If using torch.nn.DataParallel, '
            'switch to distributed training with '
            'torch.nn.parallel.DistributedDataParallel instead'
        )
        self.static_width = self.weight.infshape.width_mult()
        return self.weight.infshape.width_mult()
        
    def forward(self, x):
        x = self.output_mult * x / self.static_width
        return super(MuReadout, self).forward(x) # parent of MuReadout is Linear


class MuSharedReadout(MuReadoutWrap):
    '''`MuReadout` with weights shared with an `nn.Embedding` layer.

    Inputs:
        weight: should be weight of an `nn.Embedding` layer
        other inputs are fed to `MuReadout`
    '''
    def __init__(self, weight, bias=True, **kwargs):
        super().__init__(weight.shape[1], weight.shape[0], bias=bias, **kwargs)
        self.weight = weight
