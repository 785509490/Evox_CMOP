__all__ = ["tree_flatten", "tree_unflatten"]

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

if "Buffer" not in nn.__dict__:
    # nn.Buffer = nn.parameter.Buffer
    nn.Buffer = torch.Tensor
