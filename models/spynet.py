# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as func

from typing import List
from models.utils.flow_warp import flow_warp


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        return self.module(x)


class SpyNet(nn.Module):
    def __init__(self, model_path: str = None, return_levels: List = None):
        super(SpyNet, self).__init__()

        self.return_levels = return_levels if return_levels is not None else [5]
        self.multi_modules = nn.ModuleList([BasicModule() for _ in range(6)])

        if isinstance(model_path, str) is not None:
            dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
            model_path = os.path.join(dir_path, model_path)

            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)["params"]
            new_state_dict = {}

            for key, value in state_dict.items():
                data = key.split('.')

                data[0] = "multi_modules"
                data[2] = "module"
                new_key = ".".join(data)
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict)

        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        return (x - self.mean) / self.std

    def process(self, reference, support, width, height, width_floor, height_floor):
        flow_list = []
        reference = [self.preprocess(reference)]
        support = [self.preprocess(support)]

        for level in range(5):
            reference.insert(0, func.avg_pool2d(input=reference[0], kernel_size=2, stride=2, count_include_pad=False))
            support.insert(0, func.avg_pool2d(input=support[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = reference[0].new_zeros(
            [
                reference[0].size(0),
                2,
                int(math.floor(reference[0].size(2) / 2.0)),
                int(math.floor(reference[0].size(3) / 2.0))
            ]
        )

        for level in range(len(reference)):
            upsampled_flow = func.interpolate(input=flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0

            if upsampled_flow.size(2) != reference[level].size(2):
                upsampled_flow = func.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode="replicate")
            if upsampled_flow.size(3) != reference[level].size(3):
                upsampled_flow = func.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode="replicate")

            # print(f"shape - ref: {reference[level].shape} | "
            #       f"warp: {flow_warp(support[level], upsampled_flow.permute(0, 2, 3, 1), interpolation='bilinear', padding='border').shape} | "
            #       f"upsample: {upsampled_flow.shape}")

            features = torch.cat(
                [
                    reference[level],
                    flow_warp(
                        support[level], upsampled_flow.permute(0, 2, 3, 1), interpolation="bilinear", padding="border"
                    ),
                    upsampled_flow
                ], dim=1
            )

            flow = self.multi_modules[level](features) + upsampled_flow

            if level in self.return_levels:
                scale = 2 ** (5 - level)
                flow_output = func.interpolate(
                    input=flow, size=(height // scale, width // scale), mode="bilinear", align_corners=False
                )
                flow_output[:, 0, :, :] *= float(width // scale) / float(width_floor // scale)
                flow_output[:, 1, :, :] *= float(height // scale) / float(height_floor // scale)
                flow_list.insert(0, flow_output)

        return flow_list

    def forward(self, reference, support):
        assert (
            reference.size() == support.size()
        ), f"reference size {reference.size()} does not match support size {support.size()}!"

        height, width = reference.size(2), reference.size(3)
        width_floor = math.floor(math.ceil(width / 32.0) * 32.0)
        height_floor = math.floor(math.ceil(height / 32.0) * 32.0)

        reference, support = [func.interpolate(
            input=inputs, size=(height_floor, width_floor), mode="bilinear", align_corners=False
        ) for inputs in [reference, support]]

        flow_list = self.process(reference, support, width, height, width_floor, height_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list
