# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import numpy as np
import torch

from tqdm import tqdm

from utils.transforms import image2tensor, mod_crop


def scan_directory(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError(f"\"suffix\" must be a string or tuple of strings.")

    root = dir_path

    def _scan_directory(_dir_path, _suffix, _recursive):
        for entry in os.scandir(_dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if _suffix is None:
                    yield return_path
                elif return_path.endswith(_suffix):
                    yield return_path
            else:
                if _recursive:
                    yield from _scan_directory(entry.path, _suffix, _recursive)
                else:
                    continue

    return _scan_directory(dir_path, suffix, recursive)


def read_image_sequence(path, require_mod_crop=False, scale=1, return_image_name=False):
    if isinstance(path, list):
        image_paths = path
    else:
        image_paths = sorted(list(scan_directory(path, full_path=True)))

    images = [cv2.imread(video).astype(np.float32) / 255.0 for video in image_paths]

    if require_mod_crop:
        images = [mod_crop(image, scale) for image in images]

    images = image2tensor(images, bgr2rgb=True, is_float=True)
    images = torch.stack(images, dim=0)

    if return_image_name:
        image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        return images, image_names
    else:
        return images


def test_video(lq, model, args):
    num_frame_testing = args.tile[0]

    if num_frame_testing:
        scale = args.scale
        num_frame_overlapping = args.tile_overlap[0]
        not_overlap_border = False
        batch, dim, channel, height, width = lq.size()
        channel = channel - 1 if args.non_blind_denoising else channel
        stride = num_frame_testing - num_frame_overlapping
        dim_index_list = list(range(0, dim - num_frame_testing, stride)) + [max(0, dim - num_frame_testing)]

        estimated_output = torch.zeros(batch, dim, channel, height * scale, width * scale)
        weight_mask = torch.zeros(batch, dim, 1, 1, 1)

        for index, dim_index in enumerate(dim_index_list):
            lq_clip = lq[:, dim_index: dim_index + num_frame_testing, ...]
            out_clip = test_clip(lq_clip, model, args, index)
            out_clip_mask = torch.ones((batch, min(num_frame_testing, dim), 1, 1, 1))

            if not_overlap_border:
                if dim_index < dim_index_list[-1]:
                    out_clip[:, -num_frame_overlapping // 2:, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2:, ...] *= 0
                if dim_index > dim_index_list[0]:
                    out_clip[:, :num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, :num_frame_overlapping // 2, ...] *= 0

            estimated_output[:, dim_index:dim_index + num_frame_testing, ...].add_(out_clip)
            weight_mask[:, dim_index:dim_index + num_frame_testing, ...].add_(out_clip_mask)

        output = estimated_output.div_(weight_mask)
    else:
        window_size = args.window_size
        origin_dim = lq.size(1)
        dim_pad = (window_size[0] - origin_dim % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -dim_pad:, ...], dims=[1])], dim=1) if dim_pad else lq
        output = test_clip(lq, model, args, index=0)
        output = output[:, :origin_dim, :, :, :]

    return output


def test_clip(lq, model, args, index):
    scale = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]

    assert (
        size_patch_testing % window_size[-1] == 0
    ), "Testing patch size should be a multiple of window_size."

    if size_patch_testing:
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        batch, dim, channel, height, width = lq.size()
        channel = channel - 1 if args.non_blind_denoising else channel
        stride = size_patch_testing - overlap_size
        height_index_list = list(range(0, height - size_patch_testing, stride)) + [max(0, height - size_patch_testing)]
        width_index_list = list(range(0, width - size_patch_testing, stride)) + [max(0, width - size_patch_testing)]

        estimated_output = torch.zeros(batch, dim, channel, height * scale, width * scale)
        weight_mask = torch.zeros_like(estimated_output)

        for indices, height_index in enumerate(height_index_list):
            loop = tqdm(width_index_list)
            for _index, width_index in enumerate(loop):
                loop.set_description(
                    f"Video: {index} - "
                    f"[Height, Width]: [{indices + 1}/{len(height_index_list)}, {_index + 1}/{len(width_index_list)}]"
                )
                input_patch = lq[
                    ..., height_index:height_index + size_patch_testing, width_index:width_index + size_patch_testing
                ]
                output_patch = model(input_patch).detach().cpu()
                output_patch_mask = torch.ones_like(output_patch)

                if not_overlap_border:
                    if height_index < height_index_list[-1]:
                        output_patch[..., -overlap_size // 2:, :] *= 0
                        output_patch_mask[..., -overlap_size // 2:, :] *= 0
                    if width_index < width_index_list[-1]:
                        output_patch[..., :, -overlap_size // 2:] *= 0
                        output_patch_mask[..., :, -overlap_size // 2:] *= 0
                    if height_index > height_index_list[0]:
                        output_patch[..., :overlap_size // 2, :] *= 0
                        output_patch_mask[..., :overlap_size // 2, :] *= 0
                    if width_index > width_index_list[0]:
                        output_patch[..., :, :overlap_size // 2] *= 0
                        output_patch_mask[..., :, :overlap_size // 2] *= 0

                estimated_output[
                    ...,
                    height_index * scale:(height_index + size_patch_testing) * scale,
                    width_index * scale:(width_index + size_patch_testing) * scale
                ].add_(output_patch)
                weight_mask[
                    ...,
                    height_index * scale:(height_index + size_patch_testing) * scale,
                    width_index * scale:(width_index + size_patch_testing) * scale
                ].add_(output_patch_mask)

        output = estimated_output.div_(weight_mask)
    else:
        _, _, _, origin_height, origin_width = lq.size()
        height_pad = (window_size[1] - origin_height % window_size[1]) % window_size[1]
        width_pad = (window_size[2] - origin_width % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -height_pad:, :], dims=[3])], dim=3) if height_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -width_pad:], dims=[4])], dim=4) if width_pad else lq

        output = model(lq).detach().cpu()
        output = output[:, :, :, :origin_height * scale, :origin_width * scale]

    return output
