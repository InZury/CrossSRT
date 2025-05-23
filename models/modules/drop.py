import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, probability=None):
        super(DropPath, self).__init__()
        self.drop = probability

    def forward(self, x):
        if self.drop == 0.0 or not self.training:
            return x

        keep_probability = 1 - self.drop
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x.div(keep_probability) * random_tensor
