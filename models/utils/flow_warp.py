import torch
import torch.nn.functional as func


def flow_warp(x, flow, interpolation="bilinear", padding="zeros", align_corners=True):
    num, _, height, width = x.size()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, height, dtype=x.dtype, device=x.device),
        torch.arange(0, width, dtype=x.dtype, device=x.device),
        indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=2).float()
    grid.requires_grad = False
    warp_grid = grid + flow

    if interpolation == "nearest4":
        warp_grid_x_floor = 2.0 * torch.floor(warp_grid[:, :, :, 0]) / max(width - 1, 1) - 1.0
        warp_grid_x_ceil = 2.0 * torch.ceil(warp_grid[:, :, :, 0]) / max(width - 1, 1) - 1.0
        warp_grid_y_floor = 2.0 * torch.floor(warp_grid[:, :, :, 1]) / max(height - 1, 1) - 1.0
        warp_grid_y_ceil = 2.0 * torch.ceil(warp_grid[:, :, :, 1]) / max(height - 1, 1) - 1.0

        warp_grid_list = [[warp_grid_x_floor, warp_grid_x_ceil], [warp_grid_y_floor, warp_grid_y_ceil]]

        output = [
            func.grid_sample(
                x, torch.stack((x_grid, y_grid), dim=3),
                mode="nearest", padding_mode=padding, align_corners=align_corners
            ) for x_grid in warp_grid_list[0] for y_grid in warp_grid_list[1]
        ]

        return torch.cat(output, 1)
    else:
        warp_grid_x = 2.0 * warp_grid[:, :, :, 0] / max(width - 1, 1) - 1.0
        warp_grid_y = 2.0 * warp_grid[:, :, :, 1] / max(height - 1, 1) - 1.0
        warp_grid_scaled = torch.stack((warp_grid_x, warp_grid_y), dim=3)

        return func.grid_sample(
            x, warp_grid_scaled, mode=interpolation, padding_mode=padding, align_corners=align_corners
        )
