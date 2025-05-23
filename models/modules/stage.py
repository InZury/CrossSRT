# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from models.utils.flow_warp import flow_warp
from models.modules.deformable_conv import DCNv2PackFlowGuided
from models.modules.mlp_geglu import GEGLU
from models.modules.tmsa import TMSAG


class Stage(nn.Module):
    def __init__(
        self,
        in_dimension,
        out_dimension,
        resolution,
        depth,
        num_heads,
        window_size,
        mul_ratio=0.75,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        parallel_frames=2,
        deformable_groups=16,
        reshape=None,
        max_residual_magnitude=10,
        use_attention_checkpoint=False,
        use_feed_forward_checkpoint=False
    ):
        super(Stage, self).__init__()

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.parallel_frames = parallel_frames

        self.norm_layer = norm_layer
        self.size = 2
        self.mode = reshape

        if reshape == "none":
            self.layer_norm = nn.LayerNorm(self.out_dimension)
            self.linear = None
        elif reshape == "down":
            self.layer_norm = nn.LayerNorm(self.in_dimension * 4)
            self.linear = nn.Linear(self.in_dimension * 4, self.out_dimension)
        elif reshape == "up":
            self.layer_norm = nn.LayerNorm(self.in_dimension // 4)
            self.linear = nn.Linear(self.in_dimension // 4, self.out_dimension)

        # Residual mutual self attention
        self.front_residual_group = TMSAG(
            dim=out_dimension,
            resolution=resolution,
            depth=int(depth * mul_ratio),
            num_heads=num_heads,
            window_size=(2, window_size[1], window_size[2]),
            mutual_attention=True,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_attention_checkpoint=use_attention_checkpoint,
            use_feed_forward_checkpoint=use_feed_forward_checkpoint
        )
        self.front_linear = nn.Linear(self.out_dimension, self.out_dimension)

        # Residual only self attention
        self.back_residual_group = TMSAG(
            dim=out_dimension,
            resolution=resolution,
            depth=depth - int(depth * mul_ratio),
            num_heads=num_heads,
            window_size=window_size,
            mutual_attention=False,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_attention_checkpoint=True,
            use_feed_forward_checkpoint=use_feed_forward_checkpoint
        )
        self.back_linear = nn.Linear(self.out_dimension, self.out_dimension)

        if self.parallel_frames != 0.0:
            self.parallel_deform = DCNv2PackFlowGuided(
                self.out_dimension, self.out_dimension, 3, padding=1, deformable_groups=deformable_groups,
                max_residual_magnitude=max_residual_magnitude, parallel_frames=parallel_frames
            )
            self.parallel_fusion = GEGLU(
                self.out_dimension * (1 + 2), self.out_dimension * (1 + 2), self.out_dimension
            )

    def reshape_layer(self, mode: str, x):
        if mode == "down":
            x = x.reshape(
                x.size(0), x.size(1) * self.size * self.size, x.size(2), x.size(3) // self.size, x.size(4) // self.size
            )
        elif mode == "up":
            x = x.reshape(
                x.size(0), x.size(1) // (self.size * self.size), x.size(2), x.size(3) * self.size, x.size(4) * self.size
            )
        elif mode == "none":
            pass
        else:
            raise ValueError("Mode can be used only [\"none\", \"down\", \"up\"]!")

        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer_norm(x)
        x = x if self.linear is None else self.linear(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x

    def forward(self, x, flows_backward, flows_forward):
        x = self.reshape_layer(self.mode, x)
        x = self.front_linear(self.front_residual_group(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.back_linear(self.back_residual_group(x).transpose(1, 4)).transpose(1, 4) + x

        if self.parallel_frames != 0.0:
            x = x.transpose(1, 2)
            x_backward, x_forward = self.get_aligned_features(x, flows_backward, flows_forward, self.parallel_frames)
            x = self.parallel_fusion(
                torch.cat([x, x_backward, x_forward], dim=2).permute(0, 1, 3, 4, 2)
            ).permute(0, 4, 1, 2, 3)

        return x

    def get_aligned_features(self, x, flows_backward, flows_forward, parallel_frames):
        num = x.size(1)
        frame = parallel_frames // 2 - 1

        x_list = [torch.Tensor() for _ in range(frame + 1)]
        flow_list = [torch.Tensor() for _ in range(frame + 1)]

        # backward part
        x_backward = [torch.zeros_like(x[:, -1, ...])]

        for index in range(num - 1 + frame, frame, -1):
            x_list[0] = x[:, index - frame, ...]
            flow_list[0] = flows_backward[0][:, index - frame - 1, ...]

            if frame == 0:
                pass
            elif index == num + frame - 1:
                x_list[1] = torch.zeros_like(x[:, num - 2, ...])
                flow_list[1] = torch.zeros_like(flows_backward[1][:, num - 3, ...])

                if frame == 2:
                    x_list[2] = torch.zeros_like(x[:, -1, ...])
                    flow_list[2] = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_list[1] = x[:, index - frame + 1, ...]
                flow_list[1] = flows_backward[1][:, index - frame - 1, ...]

                if frame == 2:
                    if index == num + frame - 2:
                        x_list[2] = torch.zeros_like(x[:, -1, ...])
                        flow_list[2] = torch.zeros_like(flows_backward[2][:, -1, ...])
                    else:
                        x_list[2] = x[:, index, ...]
                        flow_list[2] = flows_forward[2][:, index - frame - 1, ...]

            x_warped_list = [
                flow_warp(x_list[frame], flow_list[frame].permute(0, 2, 3, 1), "bilinear") for _ in range(frame + 1)
            ]
            x_backward.insert(
                0, self.parallel_deform(torch.cat(x_list, dim=1), x_warped_list, x[:, index - 1 - frame], flow_list)
            )

        # forward part
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        scale = frame % 2

        for index in range(scale - 1, num + scale - 2):
            x_list[0] = x[:, index - scale + 1, ...]
            flow_list[0] = flows_forward[0][:, index - scale + 1, ...]

            if frame == 0:
                pass
            elif index == scale - 1:
                x_list[1] = torch.zeros_like(x[:, 1 - scale])
                flow_list[1] = torch.zeros_like(flows_forward[1][:, 0, ...])

                if frame == 2:
                    x_list[2] = torch.zeros_like(x[:, 0, ...])
                    flow_list[2] = torch.zeros_like(flows_backward[2][:, 0, ...])
            else:
                x_list[1] = x[:, index - scale, ...]
                flow_list[1] = flows_forward[1][:, index - scale, ...]

                if frame == 2:
                    if index == scale:
                        x_list[2] = torch.zeros_like(x[:, 0, ...])
                        flow_list[2] = torch.zeros_like(flows_backward[2][:, 0, ...])
                    else:
                        x_list[2] = x[:, index - scale, ...]
                        flow_list[2] = flows_backward[2][:, index - scale - 1, ...]

            x_warped_list = [
                flow_warp(x_list[frame], flow_list[frame].permute(0, 2, 3, 1), "bilinear") for _ in range(frame + 1)
            ]
            x_forward.append(
                self.parallel_deform(torch.cat(x_list, dim=1), x_warped_list, x[:, index - scale + 2, ...], flow_list)
            )

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]
