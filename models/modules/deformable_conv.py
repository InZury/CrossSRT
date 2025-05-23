import math

import torch
import torch.nn as nn

from torch import Tensor
from torchvision.ops import deform_conv2d


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True
    ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.is_bias = bias
        self.transposed = False
        self.output_padding = (0, )
        self.weight = nn.Parameter(Tensor(out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.init_weight()

    def init_weight(self):
        num = self.in_channels

        for key in self.kernel_size:
            num *= key

        std_value = 1. / math.sqrt(num)
        self.weight.data.uniform_(-std_value, std_value)

        if self.bias is not None:
            self.bias.data.zero_()


class ModulatedDeformConvPack(ModulatedDeformConv):
    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.stride = (self.stride, self.stride)
        self.padding = (self.padding, self.padding)
        self.dilation = (self.dilation, self.dilation)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )

        self.init_weight()

    def init_weight(self):
        super(ModulatedDeformConvPack, self).init_weight()

        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    def __init__(self, *args, **kwargs):
        self.max_residual_magnitude = kwargs.pop("max_residual_magnitude", 10)
        self.parallel_frames = kwargs.pop("parallel_frames", 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.refined_in_channels = (1 + self.parallel_frames // 2) * self.in_channels + self.parallel_frames
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.refined_in_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1)
        )
        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weight()

        if hasattr(self, "conv_offset"):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_warped_flow, x_current, flows):
        output = self.conv_offset(torch.cat(x_warped_flow + [x_current] + flows, dim=1))
        out_1, out_2, mask = torch.chunk(output, 3, dim=1)
        size = self.parallel_frames // 2

        offset = self.max_residual_magnitude * torch.tanh(torch.cat((out_1, out_2), dim=1))
        offset_list = list(torch.chunk(offset, size, dim=1))

        for index in range(size):
            offset_list[index] += flows[index].flip(1).repeat(1, offset_list[index].size(1) // 2, 1, 1)
        offset = torch.cat(offset_list, dim=1)

        mask = torch.sigmoid(mask)

        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)
