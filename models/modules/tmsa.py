import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

from functools import reduce
from torch.utils.checkpoint import checkpoint

from models.modules.window_attention import WindowAttention
from models.modules.drop import DropPath
from models.modules.mlp_geglu import GEGLU


class TMSA(nn.Module):
    # Temporal Mutual Self Attention (TMSA)

    def __init__(
        self,
        dim,
        resolution,
        num_heads,
        window_size=(6, 8, 8),
        shift_size=(0, 0, 0),
        mutual_attention=True,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.,
        activate_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_attention_checkpoint=False,
        use_feed_forward_checkpoint=False
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_attention_checkpoint = use_attention_checkpoint
        self.use_feed_forward_checkpoint = use_feed_forward_checkpoint

        self.norm_list = nn.ModuleList([norm_layer(dim) for _ in range(2)])
        self.attention = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, mutual_attention=mutual_attention
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), activate=activate_layer)

    def attention_forward(self, x, mask):
        batch, dim, height, width, channel = x.shape
        window_size, shift_size = get_window_size((dim, height, width), self.window_size, self.shift_size)

        # add cross attention

        x = self.norm_list[0](x)

        # pad feature
        left = top = front = 0
        back = (window_size[0] - dim % window_size[0]) % window_size[0]
        bottom = (window_size[1] - height % window_size[1]) % window_size[1]
        right = (window_size[2] - width % window_size[2]) % window_size[2]
        x = func.pad(x, (0, 0, left, right, top, bottom, front, back), mode="constant")

        _, pad_dim, pad_height, pad_width, _ = x.shape

        if any(index > 0 for index in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attention_mask = mask
        else:
            shifted_x = x
            attention_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attention_windows = self.attention(x_windows, mask=attention_mask)

        attention_windows = attention_windows.view(-1, *(window_size + (channel, )))
        shifted_x = window_reverse(attention_windows, window_size, batch, pad_dim, pad_height, pad_width)

        if any(index > 0 for index in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if back > 0 or right > 0 or bottom > 0:
            x = x[:, :dim, :height, :width, :]

        return self.drop_path(x)

    def mlp_forward(self, x):
        return self.drop_path(self.mlp(self.norm_list[1](x)))

    def forward(self, x, mask):
        if self.use_attention_checkpoint:
            x += checkpoint(self.attention_forward, x, mask, use_reentrant=False)
        else:
            x += self.attention_forward(x, mask)

        if self.use_feed_forward_checkpoint:
            x += checkpoint(self.mlp_forward, x, use_reentrant=False)
        else:
            x += self.mlp_forward(x)

        return x


class TMSAG(nn.Module):
    # Group by TMSA

    def __init__(
            self,
            dim,
            resolution,
            depth,
            num_heads,
            window_size=None,
            shift_size=None,
            mutual_attention=True,
            mlp_ratio=2.,
            qkv_bias=False,
            qk_scale=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            use_attention_checkpoint=False,
            use_feed_forward_checkpoint=False
    ):
        super().__init__()
        self.input_resolution = resolution
        self.window_size = window_size if window_size is not None else [6, 8, 8]
        self.shift_size = list(index // 2 for index in window_size) if shift_size is None else shift_size

        self.blocks = nn.ModuleList([
            TMSA(
                dim=dim,
                resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if index % 2 == 0 else self.shift_size,
                mutual_attention=mutual_attention,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_attention_checkpoint=use_attention_checkpoint,
                use_feed_forward_checkpoint=use_feed_forward_checkpoint
            ) for index in range(depth)
        ])

    def forward(self, x):
        batch, channel, dim, height, width = x.shape
        window_size, shift_size = get_window_size((dim, height, width), self.window_size, self.shift_size)
        x = x.permute(0, 2, 3, 4, 1)
        pad_dim, pad_height, pad_width = [
            int(np.ceil(shape / window_size[index])) * window_size[index]
            for shape, index in zip([dim, height, width], [0, 1, 2])
        ]
        attention_mask = compute_mask(pad_dim, pad_height, pad_width, window_size, shift_size, x.device)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = x.view(batch, dim, height, width, -1)
        x = x.permute(0, 4, 1, 2, 3)

        return x


class RTMSA(nn.Module):
    # Residual TMSA. Only used in stage 8.

    def __init__(
        self,
        dim,
        resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        use_attention_checkpoint=False,
        use_feed_forward_checkpoint=False
    ):
        super(RTMSA, self).__init__()

        self.dim = dim
        self.resolution = resolution
        self.residual_group = TMSAG(
            dim=dim,
            resolution=resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mutual_attention=False,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_attention_checkpoint=use_attention_checkpoint,
            use_feed_forward_checkpoint=use_feed_forward_checkpoint
        )

        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)

    if shift_size is not None:
        use_shift_size = list(shift_size)
    else:
        use_shift_size = None

    for index in range(len(x_size)):
        if x_size[index] <= window_size[index]:
            use_window_size[index] = x_size[index]

            if shift_size is not None:
                use_shift_size[index] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def window_partition(x, window_size):
    batch, dim, height, width, channel = x.shape
    x = x.view(batch, dim // window_size[0], window_size[0],
               height // window_size[1], window_size[1], width // window_size[2], window_size[2], channel)

    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, reduce(lambda shape, current: shape * current, window_size), channel
    )


def window_reverse(windows, window_size, batch, dim, height, width):
    x = windows.view(
        batch, dim // window_size[0], height // window_size[1], width // window_size[2],
        window_size[0], window_size[1], window_size[2], -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(batch, dim, height, width, -1)

    return x


def compute_mask(dim, height, width, window_size, shift_size, device):
    image_mask = torch.zeros((1, dim, height, width, 1), device=device)
    count = 0

    for dim in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for height in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for width in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
                image_mask[:, dim, height, width, :] = count
                count += 1

    mask_windows = window_partition(image_mask, window_size).squeeze(-1)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(
        attention_mask != 0, float(-100.0)
    ).masked_fill(attention_mask == 0, float(0.0))

    return attention_mask
