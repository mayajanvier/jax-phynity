import collections.abc as collections
import functools
import logging
import random
import warnings
import os
import sys
import jax 
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, Type, TypeVar, Union, cast

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn import init
# from torch import Tensor
# from torch.nn import Parameter
# from datetime import datetime

# class Logger(object):
#     "Lumberjack class - duplicates sys.stdout to a log file and it's okay"
#     def __init__(self, filename, mode="a"):
#         self.stdout = sys.stdout
#         self.file = open(filename, mode)
#         sys.stdout = self

#     def __del__(self):
#         self.close()

#     def __enter__(self):
#         pass

#     def __exit__(self, *args):
#         self.close()

#     def write(self, message):
#         self.stdout.write(message)
#         self.file.write(message)

#     def flush(self):
#         self.stdout.flush()
#         self.file.flush()
#         os.fsync(self.file.fileno())

#     def close(self):
#         if self.stdout != None:
#             sys.stdout = self.stdout
#             self.stdout = None

#         if self.file != None:
#             self.file.close()
#             self.file = None

# def set_requires_grad(nets, requires_grad=False):
#     if not isinstance(nets, list):
#         nets = [nets]
#     for net in nets:
#         if net is not None:
#             for param in net.parameters():
#                 param.requires_grad = requires_grad

# def fix_seed(seed):
#     import numpy as np
#     import torch
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     torch.cuda.manual_seed_all(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)


# def pretty_wrap(text, title=None, width=80):
#     table = pt.PrettyTable(
#         header=title is not None,
#     )
#     table.field_names = [title]
#     for t in text.split('\n'):
#         for i in range(0, len(t), width):
#             table.add_row([t[i: i + width]])

#     return table

# def make_basedir(root, timestamp=None, attempts=5):
#     """Takes 5 shots at creating a folder from root,
#     adding timestamp if desired.
#     """
#     for i in range(attempts):
#         basedir = root
#         if timestamp is None:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#             basedir = os.path.join(basedir, timestamp)
#         try:
#             os.makedirs(basedir)
#             return basedir
#         except:
#             sleep(0.01)
#     raise FileExistsError(root)

# ################################################################################
# # Adapted from http://
# ################################################################################

# def l2normalize(v, eps=1e-12):
#     return v / (v.norm() + eps)

# class CalculateNorm:
#     def __init__(self, module, power_iterations=5):
#         self.module = module
#         assert isinstance(module, nn.ModuleList)
#         self.power_iterations = power_iterations
#         self._make_params()

#     def calculate_spectral_norm(self):
#         # Adapted to complex weights
#         sigmas = [0. for i in range(len(self.module))]
#         for i, module in enumerate(self.module):
#             for name, w in module.named_parameters():
#                 if name.find('bias') == -1 and name.find('beta') == -1:
#                     u = self.u[f'{i},{name}']
#                     v = self.v[f'{i},{name}']

#                     height = w.data.shape[0]
#                     for _ in range(self.power_iterations):
#                         v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
#                         u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

#                     sigma = torch.conj(u).dot(w.view(height, -1).mv(v))
#                     if torch.is_complex(sigma):
#                         sigmas[i] = sigmas[i] + sigma.real ** 2
#                     else:
#                         sigmas[i] = sigmas[i] + sigma ** 2
#         return torch.stack(sigmas)

#     def calculate_frobenius_norm(self):
#         # Only used for linear case
#         sigmas = [0. for i in range(len(self.module))]
#         for i, module in enumerate(self.module):
#             for name, w in module.named_parameters():
#                 if name.find('bias') == -1 and name.find('beta') == -1:
#                     sigmas[i] = sigmas[i] + torch.norm(w)
#         return torch.stack(sigmas)

#     def _make_params(self):
#         self.u, self.v = dict(), dict()
#         for i, module in enumerate(self.module):
#             for name, w in module.named_parameters():
#                 if name.find('bias') == -1 and name.find('beta') == -1:
#                     height = w.data.shape[0]
#                     width = w.view(height, -1).data.shape[1]

#                     u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#                     v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
#                     u.data = l2normalize(u.data)
#                     v.data = l2normalize(v.data)

#                     self.u[f'{i},{name}'] = u
#                     self.v[f'{i},{name}'] = v

# def init_weights(net, init_type='normal', init_gain=0.02):
#     print(net.weight.data)
#     def init_func(m):
#         classname = m.__class__.__name__
#         print(classname)
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'orthogonal':
#                 print("orthogonal")
#                 m.weight.data = jax.nn.initializers.orthogonal(gain=init_gain)
#                 #init.orthogonal_(m.weight.data, gain=init_gain)
#             elif init_type == 'default':
#                 pass
#             if hasattr(m, 'bias') and m.bias is not None:
#                 print("bias")
#                 m.bias.data = jax.nn.initializers.constant(0.0)
#     #net.apply(init_func)
#     init_func(net)
#     print(net.weight.data)

def orthogonal_init(key: jax.random.PRNGKey, shape: tuple, gain: float = 1.0) -> jax.Array:
    """Applies orthogonal initialization to a non-square matrix."""
    out_features, in_features = shape
    
    if out_features >= in_features:  # More output features than input features
        mat = jax.random.normal(key, (out_features, in_features))
        q, r = jnp.linalg.qr(mat)  # QR decomposition to get orthogonal columns
        d = jnp.sign(jnp.diag(r))  # Normalize the signs of the diagonal of r
        return gain * q * d  # Orthogonal matrix scaled by gain
    else:  # More input features than output features
        mat = jax.random.normal(key, (in_features, out_features))
        q, r = jnp.linalg.qr(mat.T)  # Transpose to get orthogonal rows
        d = jnp.sign(jnp.diag(r))  # Normalize the signs of the diagonal of r
        return gain * q.T * d  # Return transposed matrix scaled by gain

def init_linear_weight(model, init_fn, key, init_gain=0.2):
    """Applies a given weight initialization function to all eqx.nn.Linear layers in a model."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)  # Check if it's a Linear layer
    # Extract weights and biases from the model
    get_weights = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]
    
    get_biases = lambda m: [x.bias 
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    
    weights = get_weights(model)  
    biases = get_biases(model)

    # Initialization for weights
    new_weights = [init_fn(subkey, weight.shape, init_gain)  
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    # Zero initialization for biases
    new_biases = [jnp.zeros_like(bias) for bias in biases]

    # Update the model
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)   
    return new_model


if __name__ == '__main__':
    from networks import MLP
    init_key, key = jax.random.split(jax.random.PRNGKey(0))
    model_aug = MLP(key=key, state_c=2, hidden=200)
    model_aug = init_linear_weight(model_aug, orthogonal_init, key=init_key, init_gain=0.2)
    print(model_aug)
    w = model_aug.layers[2].weight
    print((w.T @ w)/(0.2**2)) # Identity matrix
    print(model_aug.layers[2].bias) # Zero bias
