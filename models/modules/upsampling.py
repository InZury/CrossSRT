import math
import torch.nn as nn


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feature):
        class TransposeDim12(nn.Module):
            def __init__(self):
                super().__init__()

            @classmethod
            def forward(cls, x):
                return x.transpose(1, 2)

        modules = []

        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv3d(num_feature, 4 * num_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                modules.append(TransposeDim12())
                modules.append(nn.PixelShuffle(2))
                modules.append(TransposeDim12())
                modules.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

            modules.append(nn.Conv3d(num_feature, num_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            modules.append(nn.Conv3d(num_feature, 9 * num_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            modules.append(TransposeDim12())
            modules.append(nn.PixelShuffle(3))
            modules.append(TransposeDim12())
            modules.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            modules.append(nn.Conv3d(num_feature, num_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")

        super(Upsample, self).__init__(*modules)
