# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as func

from itertools import chain

from models.spynet import SpyNet
from models.modules.stage import Stage
from models.modules.tmsa import RTMSA
from models.modules.upsampling import Upsample
from models.utils.flow_warp import flow_warp


class CSRT(nn.Module):
    def __init__(
        self,
        config,
        layer_norm=nn.LayerNorm
    ):
        super().__init__()

        self.upscale = config.parameters.upscale
        self.in_channels = config.parameters.in_channels
        self.out_channels = config.parameters.out_channels
        self.parallel_frame = config.parameters.parallel_frame
        self.drop_path = config.parameters.drop_path
        self.num_heads = config.parameters.num_heads
        self.mul_ratio = config.parameters.mul_ratio
        self.mlp_ratio = config.parameters.mlp_ratio
        self.deformable_groups = config.parameters.deformable_groups

        self.qkv_bias = config.parameters.qkv_bias
        self.qk_scale = None if config.parameters.qk_scale == "None" else config.parameters.qk_scale

        self.realign_all_flows = config.parameters.realign_all_flows
        self.non_blind_denoising = config.parameters.non_blind_denoising
        self.use_attention_checkpoint = config.parameters.use_attention_checkpoint
        self.use_feed_forward_checkpoint = config.parameters.use_feed_forward_checkpoint
        self.except_attention_checkpoint = config.parameters.except_attention_checkpoint
        self.except_feed_forward_checkpoint = config.parameters.except_feed_forward_checkpoint

        self.image_size = config.parameters.image_size
        self.window_size = config.parameters.window_size
        self.depths = config.parameters.depths
        self.dimensions = config.parameters.dimensions
        self.flatten_layer = config.parameters.flatten_layer

        if self.parallel_frame != 0.0:
            if self.non_blind_denoising:
                resize = self.in_channels * (1 + 2 * 4) + 1
            else:
                resize = self.in_channels * (1 + 2 * 4)
        else:
            resize = self.in_channels

        # align layer
        self.first_conv = nn.Conv3d(resize, self.dimensions[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # csrt module
        if self.parallel_frame != 0.0:
            pretrained_path = config.pretrained.model_path
            self.spynet = SpyNet(pretrained_path, [2, 3, 4, 5])

        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, sum(self.depths))]
        reshapes = ["none", "down", "down", "down", "up", "up", "up"]
        scales = [1, 2, 4, 8, 4, 2, 1]
        use_attentions_checkpoint = [
            False if i in self.except_attention_checkpoint else self.use_attention_checkpoint
            for i in range(len(self.depths))
        ]
        use_feed_forwards_checkpoint = [
            False if i in self.except_feed_forward_checkpoint else self.use_feed_forward_checkpoint
            for i in range(len(self.depths))
        ]

        for index in range(7):
            setattr(
                self, f"stage{index+1}",
                Stage(
                    in_dimension=self.dimensions[index - 1],
                    out_dimension=self.dimensions[index],
                    resolution=(
                        self.image_size[0], self.image_size[1] // scales[index], self.image_size[2] // scales[index]
                    ),
                    depth=self.depths[index],
                    num_heads=self.num_heads,
                    mul_ratio=self.mul_ratio,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    drop_path=drop_path[sum(self.depths[:index]):sum(self.depths[:index + 1])],
                    norm_layer=layer_norm,
                    parallel_frames=self.parallel_frame,
                    deformable_groups=self.deformable_groups,
                    reshape=reshapes[index],
                    max_residual_magnitude=10 / scales[index],
                    use_attention_checkpoint=use_attentions_checkpoint[index],
                    use_feed_forward_checkpoint=use_feed_forwards_checkpoint[index]
                )
            )

        self.pre_stage8 = nn.Sequential(
            nn.LayerNorm(self.dimensions[6]),
            nn.Linear(self.dimensions[6], self.dimensions[7])
        )

        self.stage8 = nn.ModuleList([])

        for index in range(7, len(self.depths)):
            self.stage8.append(
                RTMSA(
                    dim=self.dimensions[index],
                    resolution=self.image_size,
                    depth=self.depths[index],
                    num_heads=self.num_heads,
                    window_size=[1, self.window_size[1], self.window_size[2]]
                    if index in self.flatten_layer else self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    drop_path=drop_path[sum(self.depths[:index]):sum(self.depths[:index + 1])],
                    norm_layer=layer_norm,
                    use_attention_checkpoint=use_attentions_checkpoint[index],
                    use_feed_forward_checkpoint=use_feed_forwards_checkpoint[index]
                )
            )

        self.norm = layer_norm(self.dimensions[-1])
        self.second_conv = nn.Linear(self.dimensions[-1], self.dimensions[0])

        if self.parallel_frame != 0.0:
            if self.upscale == 1:
                self.last_conv = nn.Conv3d(
                    self.dimensions[0], self.out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                )
            else:
                num_feature = 64
                self.upsample_conv = nn.Sequential(
                    nn.Conv3d(self.dimensions[0], num_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.LeakyReLU(inplace=True)
                )
                self.upsample = Upsample(self.upscale, num_feature)
                self.last_conv = nn.Conv3d(num_feature, self.out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        else:
            num_feature = 64
            self.linear_fusion = nn.Conv2d(
                self.dimensions[0] * self.image_size[0], num_feature, kernel_size=1, stride=1
            )
            self.last_conv = nn.Conv2d(num_feature, self.out_channels, kernel_size=7, stride=1, padding=0)

    def get_flow_frames(self, x):
        flows_backward, flows_forward = [], []

        batch, num, channel, height, width = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, channel, height, width)
        x_2 = x[:, 1:, :, :, :].reshape(-1, channel, height, width)

        flows_backward.append(self.spynet(x_1, x_2))
        flows_backward[0] = [
            flow.view(batch, num - 1, 2, height // (2 ** index), width // (2 ** index))
            for flow, index in zip(flows_backward[0], range(4))
        ]

        flows_forward.append(self.spynet(x_2, x_1))
        flows_forward[0] = [
            flow.view(batch, num - 1, 2, height // (2 ** index), width // (2 ** index))
            for flow, index in zip(flows_forward[0], range(4))
        ]

        for indices in range(self.parallel_frame // 2 - 1):
            dim = flows_forward[indices][0].shape[1]
            temporal_flows_backward = []
            temporal_flows_forward = []

            for flows_1, flows_2 in zip(flows_backward[indices-1], flows_backward[indices]):
                flow_list = []

                for index in range(dim - 1, 0, -1):
                    flow_1 = flows_2[:, index - 1, :, :, :]
                    flow_2 = flows_1[:, index + indices, :, :, :]
                    flow_list.insert(0, flow_1 + flow_warp(flow_2, flow_1.permute(0, 2, 3, 1)))

                temporal_flows_backward.append(torch.stack(flow_list, 1))

            for flows_1, flows_2 in zip(flows_forward[indices-1], flows_forward[indices]):
                flow_list = []

                for index in range(indices + 1, dim + indices):
                    flow_1 = flows_2[:, index - indices, :, :, :]
                    flow_2 = flows_1[:, index - indices - 1, :, :, :]
                    flow_list.append(flow_1 + flow_warp(flow_2, flow_1.permute(0, 2, 3, 1)))

                temporal_flows_forward.append(torch.stack(flow_list, 1))

            flows_backward.append(temporal_flows_backward)
            flows_forward.append(temporal_flows_forward)

        return list(chain.from_iterable(flows_backward)), list(chain.from_iterable(flows_forward))

    @classmethod
    def get_aligned_image_frames(cls, x, flows_backward, flows_forward):
        num = x.size(1)

        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]

        for index in range(num - 1, 0, -1):
            x_1 = x[:, index, ...]
            flow = flows_backward[:, index - 1, ...]
            x_backward.insert(0, flow_warp(x_1, flow.permute(0, 2, 3, 1), "nearest4"))

        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]

        for index in range(0, num - 1):
            x_1 = x[:, index, ...]
            flow = flows_forward[:, index, ...]
            x_forward.append(flow_warp(x_1, flow.permute(0, 2, 3, 1), "nearest4"))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def forward_features(self, x, flows_backward, flows_forward):
        x_1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x_2 = self.stage2(x_1, flows_backward[1::4], flows_forward[1::4])
        x_3 = self.stage3(x_2, flows_backward[2::4], flows_forward[2::4])
        x_4 = self.stage4(x_3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x_4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x_3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x_2, flows_backward[0::4], flows_forward[0::4])
        x = x + x_1

        x = x.permute(0, 2, 3, 4, 1)
        x = self.pre_stage8(x)
        x = x.permute(0, 4, 1, 2, 3)

        for layer in self.stage8:
            x = layer(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x

    @classmethod
    def reflection_padding_2d(cls, x, pad=1):
        x = torch.cat([torch.flip(x[:, :, 1:pad + 1, :], [2]), x, torch.flip(x[:, :, -pad - 1:-1, :], [2])], 2)
        x = torch.cat([torch.flip(x[:, :, :, 1:pad + 1], [3]), x, torch.flip(x[:, :, :, -pad - 1:-1], [3])], 3)

        return x

    def forward(self, x):
        if self.parallel_frame:
            if self.non_blind_denoising:
                x = x[:, :, :self.in_channels, :, :]

            x_lq = x.clone()

            flows_backward, flows_forward = self.get_flow_frames(x)

            x_backward, x_forward = self.get_aligned_image_frames(x, flows_backward[0], flows_forward[0])
            x = torch.cat([x, x_backward, x_forward], 2)

            if self.non_blind_denoising:
                x = torch.cat([x, x[:, :, self.in_channels:, :, :]], 2)     # cat(x, noise_level_map)

            if self.upscale == 1:
                x = self.first_conv(x.transpose(1, 2))
                x += self.second_conv(
                    self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)
                ).transpose(1, 4)
                x = self.last_conv(x).transpose(1, 2)

                return x + x_lq
            else:
                x = self.first_conv(x.transpose(1, 2))
                x += self.second_conv(
                    self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)
                ).transpose(1, 4)
                x = self.last_conv(self.upsample(self.upsample_conv(x))).transpose(1, 2)
                _, _, channel, height, width = x.shape

                return x + func.interpolate(x_lq, size=(channel, height, width), mode="trilinear", align_corners=False)
        else:
            x_mean = x.mean([1, 3, 4], keepdim=True)
            x -= x_mean

            x = self.first_conv(x.transpose(1, 2))
            x += self.second_conv(
                self.forward_features(x, [], []).transpose(1, 4)
            ).transpose(1, 4)

            x = torch.cat(torch.unbind(x, 2), 1)
            x = self.last_conv(self.reflection_padding_2d(func.leaky_relu(self.linear_fusion(x), 0.2), pad=3))
            x = torch.stack(torch.split(x, dim=1, split_size_or_sections=3), 1)

            return x + x_mean
