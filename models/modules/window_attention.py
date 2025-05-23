import math
import warnings
import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        mutual_attention=True
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.mutual_attention = mutual_attention

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads
            )
        )
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim)

        if self.mutual_attention:
            self.register_buffer(
                "position_bias", self.get_sine_position_encoding(window_size[1:], dim // 2, normalize=True)
            )
            self.qkv_mutual = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.projection = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self._trunc_normal(self.relative_position_bias_table, std=.02)

    @classmethod
    def get_position_index(cls, window_size):
        coordinate = [torch.arange(window_size[index]) for index in range(3)]               # 3, dim, height, width
        coordination = torch.stack(torch.meshgrid(*coordinate, indexing="ij"))
        coordination = torch.flatten(coordination, 1)                             # 3, dim*height*width

        # 3, dim*height*width, dim*height*width
        relative_coordination = coordination[:, :, None] - coordination[:, None, :]

        # dim*height*width, dim*height*width, 3
        relative_coordination = relative_coordination.permute(1, 2, 0).contiguous()

        for index in range(3):
            relative_coordination[:, :, index] += window_size[index] - 1                    # shift window

        relative_coordination[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coordination[:, :, 1] *= (2 * window_size[2] - 1)

        return relative_coordination.sum(-1)                                      # dim*height*width, dim*height*width

    @classmethod
    def get_sine_position_encoding(cls, size, num_position_features=64, temperature=10000, normalize=False, scale=None):
        if scale is None:
            scale = 2 * math.pi
        else:
            if normalize is False:
                raise ValueError("normalize should be True if scale is passed!")

        not_mask = torch.ones([1, size[0], size[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if normalize:
            epsilon = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * scale

        dim_temperature = torch.arange(num_position_features, dtype=torch.float32)
        dim_temperature = temperature ** (2 * (dim_temperature // 2) / num_position_features)

        position = [embed[:, :, :, None] / dim_temperature for embed in [y_embed, x_embed]]
        position = [
            torch.stack((position[index][:, :, :, 0::2].sin(), position[index][:, :, :, 1::2].cos()), dim=4).flatten(3)
            for index in range(2)
        ]
        position_embed = torch.cat(position, dim=3).permute(0, 3, 1, 2)

        return position_embed.flatten(2).permute(0, 2, 1).contiguous()

    @classmethod
    def _trunc_normal(cls, tensor, mean=0., std=1., alpha=-2., beta=2.):
        # no gradient trunc normal

        def normal_cumulative_distribution_function(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < alpha - 2 * std) or (mean > beta + 2 * std):
            warnings.warn(
                "Mean is more than 2 std from [alpha, beta] in nn.init.trunc_normal_. "
                "The distribution of value may be incorrect.", stacklevel=2
            )

        with torch.no_grad():
            low = normal_cumulative_distribution_function((alpha - mean) / std)
            high = normal_cumulative_distribution_function((beta - mean) / std)

            tensor.uniform_(2 * low - 1, 2 * high - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp(min=alpha, max=beta)

            return tensor

    def attention(self, query, key, value, mask, x_shape, relative_position_encoding=True):
        batch, num, channel = x_shape
        attention = (query * self.scale) @ key.transpose(-2, -1)

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:num, :num].reshape(-1)
            ].reshape(num, num, -1)
            attention += relative_position_bias.permute(2, 0, 1).unsqueeze(0)

        if mask is None:
            attention = self.softmax(attention)
        else:
            num_window = mask.shape[0]
            attention = attention.view(batch // num_window, num_window, self.num_heads, num, num)
            attention += mask[:, :num, :num].unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, num, num)
            attention = self.softmax(attention)

        return (attention @ value).transpose(1, 2).reshape(batch, num, channel)

    def forward(self, x, mask=None):
        batch, num, channel = x.shape
        qkv = self.qkv(x).reshape(batch, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        output = self.attention(query, key, value, mask, x_shape=(batch, num, channel), relative_position_encoding=True)

        if self.mutual_attention:
            qkv = self.qkv_mutual(
                x + self.position_bias.repeat(1, 2, 1)
            ).reshape(batch, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
            query, key, value = [torch.chunk(qkv[index], 2, dim=2) for index in range(3)]

            aligned_x = [
                self.attention(
                    query[1 - index], key[index], value[index], mask, (batch, num // 2, channel),
                    relative_position_encoding=False
                ) for index in range(2)
            ]
            output = torch.cat([torch.cat(aligned_x, dim=1), output], dim=2)

        return self.projection(output)
