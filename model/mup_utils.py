import torch
from collections import defaultdict

'''
Adopted from https://github.com/microsoft/mup
Standalone implementation so that we don't need to write meta-data to parameters.
'''

class MuReadoutS(torch.nn.Linear):
    '''standalone version of muP MuReadout
    https://github.com/microsoft/mup
    Removes inplace op allowing compile, base_fan_in is required
    '''
    def __init__(self, *args, readout_zero_init=False, output_mult=1.0, base_fan_in=None, **kwargs):
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        super().__init__(*args, **kwargs)
        _, fan_in = self.weight.shape
        self.width_mult = fan_in / base_fan_in
    
    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.weight.data[:] = 0
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            super().reset_parameters()
            
    def forward(self, x):
        x = self.output_mult * x / self.width_mult
        return super().forward(x) # parent of MuReadoutS is Linear


class MuSharedReadoutS(MuReadoutS):
    '''standalone version of muP MuSharedReadoutS
    `MuReadout` with weights shared with an `nn.Embedding` layer.
    Inputs:
        weight: should be weight of an `nn.Embedding` layer
        other inputs are fed to `MuReadoutS`
    '''
    def __init__(self, weight, bias=True, **kwargs):
        super().__init__(*weight.shape, bias=bias, **kwargs)
        self.weight = weight



def adam_mu_param_groups(named_parameters, param_base_shape_and_mult, **kwargs):
    """
    Creates new parameter groups with scaled learning rates and weight decays
    based on a dictionary containing base shapes and width multipliers.
    NOTE: can't take param_groups as input; just one group from scratch.
        TODO: Function to handle param groups

    Args:
        named_parameters: An iterator over (name, parameter) pairs from the model.
        param_base_shape_and_mult: A dictionary mapping parameter names to a list 
                                containing their base shape and width multiplier.
        **kwargs: Additional keyword arguments to be passed to the optimizer.

    Returns:
        A list of new parameter groups with scaled learning rates and weight decays.
    """
    new_param_groups = []
    # Process potential existing param groups (if any)

    param_group = {'names': [], 'params': []}
    for name, param  in named_parameters:
        param_group['names'].append(name)
        param_group['params'].append(param)
    param_group['lr'] = kwargs['lr']
    param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
        
    def new_group():
        new_g = {k: v for k, v in param_group.items() if k != 'params'}
        new_g['params'] = []
        return new_g

    matrix_like_p = defaultdict(new_group)
    vector_like_p = new_group()

    for name, p in zip(param_group['names'],param_group['params']):
        match = next((key for key in param_base_shape_and_mult if key in name + '.weight'), None)
        if match is None:
            match = next((
                key for key in param_base_shape_and_mult
                if key in name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "")
                + ".weight"), None)
        # shape_and_mult = param_base_shape_and_mult.get(name)
        if match:  # Parameter has a base shape
            shape_and_mult = param_base_shape_and_mult[match]
            ninf = len(shape_and_mult[0]) - shape_and_mult[0].count(None)  # Count infinite dimensions
            if ninf == 2:  # Matrix-like
                matrix_like_p[shape_and_mult[1]]['params'].append(p)
            elif ninf <= 1:  # Vector-like
                vector_like_p['params'].append(p)
            else:
                raise NotImplementedError('more than 2 inf dimensions')
        else:
            raise RuntimeError(f'{name} not in param_base_shape_and_mult')

    for width_mult, group in matrix_like_p.items():
        group['lr'] /= width_mult
        if not kwargs.get('decoupled_wd', False):
            group['weight_decay'] *= width_mult
    new_param_groups = list(matrix_like_p.values()) + [vector_like_p]

    return new_param_groups

from collections import defaultdict

from mup import MuReadout
import torch


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


class MuReadoutS(torch.nn.Linear):
    '''standalone version of muP MuReadout
    https://github.com/microsoft/mup
    Removes inplace op allowing compile, base_fan_in is required
    '''
    def __init__(self, *args, readout_zero_init=False, output_mult=1.0, base_fan_in=None, **kwargs):
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        super().__init__(*args, **kwargs)
        _, fan_in = self.weight.shape
        self.width_mult = fan_in / base_fan_in

    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.weight.data[:] = 0
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            super().reset_parameters()

    def forward(self, x):
        x = self.output_mult * x / self.width_mult
        return super().forward(x) # parent of MuReadoutS is Linear


class MuSharedReadoutS(MuReadoutS):
    '''standalone version of muP MuSharedReadoutS
    `MuReadout` with weights shared with an `nn.Embedding` layer.
    Inputs:
        weight: should be weight of an `nn.Embedding` layer
        other inputs are fed to `MuReadoutS`
    '''
    def __init__(self, weight, bias=True, **kwargs):
        super().__init__(*weight.shape, bias=bias, **kwargs)
        self.weight = weight


def adam_mu_param_groups(named_parameters, param_base_shape_and_mult, **kwargs):
    """
    Creates new parameter groups with scaled learning rates and weight decays
    based on a dictionary containing base shapes and width multipliers.
    NOTE: can't take param_groups as input; just one group from scratch.
        TODO: Function to handle param groups

    Args:
        named_parameters: An iterator over (name, parameter) pairs from the model.
        param_base_shape_and_mult: A dictionary mapping parameter names to a list
                                containing their base shape and width multiplier.
        **kwargs: Additional keyword arguments to be passed to the optimizer.

    Returns:
        A list of new parameter groups with scaled learning rates and weight decays.
    """
    new_param_groups = []
    # Process potential existing param groups (if any)

    param_group = {'names': [], 'params': []}
    for name, param  in named_parameters:
        param_group['names'].append(name)
        param_group['params'].append(param)
    param_group['lr'] = kwargs['lr']
    param_group['weight_decay'] = kwargs.get('weight_decay', 0.)

    def new_group():
        new_g = {k: v for k, v in param_group.items() if k != 'params'}
        new_g['params'] = []
        return new_g

    matrix_like_p = defaultdict(new_group)
    vector_like_p = new_group()

    for name, p in zip(param_group['names'],param_group['params']):
        match = next((key for key in param_base_shape_and_mult if key in name + '.weight'), None)
        if match is None:
            match = next((
                key for key in param_base_shape_and_mult
                if key in name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "")
                + ".weight"), None)
        # shape_and_mult = param_base_shape_and_mult.get(name)
        if match:  # Parameter has a base shape
            shape_and_mult = param_base_shape_and_mult[match]
            ninf = len(shape_and_mult[0]) - shape_and_mult[0].count(None)  # Count infinite dimensions
            if ninf == 2:  # Matrix-like
                matrix_like_p[shape_and_mult[1]]['params'].append(p)
            elif ninf <= 1:  # Vector-like
                vector_like_p['params'].append(p)
            else:
                raise NotImplementedError('more than 2 inf dimensions')
        else:
            raise RuntimeError(f'{name} not in param_base_shape_and_mult')

    for width_mult, group in matrix_like_p.items():
        group['lr'] /= width_mult
        if not kwargs.get('decoupled_wd', False):
            group['weight_decay'] *= width_mult
    new_param_groups = list(matrix_like_p.values()) + [vector_like_p]

    return new_param_groups
