# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.utils.data as data

from glob import glob

from utils.video import scan_directory, read_image_sequence


class VideoRecurrentTestDataset(data.Dataset):
    def __init__(self, kwargs):
        super(VideoRecurrentTestDataset, self).__init__()

        self.cache_data = kwargs["cache_data"]
        self.lq_root = kwargs["lq_data"]
        self.gt_root = kwargs["gt_data"]
        self.data_info = {
            "lq_path": [],
            "gt_path": [],
            "folder": [],
            "index": [],
            "border": []
        }

        self.lq_images, self.gt_images = {}, {}
        lq_subfolders = sorted(glob(os.path.join(self.lq_root, '*')))
        gt_subfolders = sorted(glob(os.path.join(self.gt_root, '*')))

        for lq_subfolder, gt_subfolder in zip(lq_subfolders, gt_subfolders):
            subfolder_name = os.path.basename(lq_subfolder)
            lq_image_paths = sorted(list(scan_directory(lq_subfolder, full_path=True)))
            gt_image_paths = sorted(list(scan_directory(gt_subfolder, full_path=True)))

            max_index = len(lq_image_paths)
            assert (
                    max_index == len(gt_image_paths)
            ), f"Different number of images in lq ({max_index}) and gt folders ({len(gt_image_paths)})."

            self.data_info["lq_path"].extend(lq_image_paths)
            self.data_info["gt_path"].extend(gt_image_paths)
            self.data_info["folder"].extend([subfolder_name] * max_index)

            for index in range(max_index):
                self.data_info["index"].append(f"{index}/{max_index}")

            border_line = [0] * max_index

            for index in range(kwargs["num_frame"] // 2):
                border_line[index] = 1
                border_line[max_index - index - 1] = 1

            self.data_info["border"].extend(border_line)

            if self.cache_data:
                print(f"Cache {subfolder_name} for VideoTestDataset...")
                self.lq_images[subfolder_name] = read_image_sequence(lq_image_paths)
                self.gt_images[subfolder_name] = read_image_sequence(gt_image_paths)
            else:
                self.lq_images[subfolder_name] = lq_image_paths
                self.gt_images[subfolder_name] = gt_image_paths

        self.folders = sorted(list(set(self.data_info["folder"])))
        self.sigma = kwargs["sigma"] / 255.0 if "sigma" in kwargs else 0

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.sigma:
            if self.cache_data:
                gt_images = self.gt_images[folder]
            else:
                gt_images = read_image_sequence(self.gt_images[folder])

            torch.manual_seed(0)
            noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
            noise = torch.normal(mean=0, std=noise_level.expand_as(gt_images))
            lq_images = gt_images + noise
            step, _, height, width = lq_images.shape
            lq_images = torch.cat([lq_images, noise_level.expand(step, 1, height, width)], dim=1)
        else:
            if self.cache_data:
                lq_images = self.lq_images[folder]
                gt_images = self.gt_images[folder]
            else:
                lq_images = read_image_sequence(self.lq_images[folder])
                gt_images = read_image_sequence(self.gt_images[folder])

        return {
            "low": lq_images,
            "high": gt_images,
            "folder": folder,
            "lq_path": self.lq_images[folder]
        }

    def __len__(self):
        return len(self.folders)


class SingleVideoRecurrentTestDataset(data.Dataset):
    def __init__(self, kwargs):
        super(SingleVideoRecurrentTestDataset, self).__init__()

        self.config = kwargs
        self.cache_data = kwargs["cache_data"]
        self.lq_root = kwargs["lq_data"]
        self.data_info = {
            "lq_path": [],
            "folder": [],
            "index": [],
            "border": []
        }

        self.lq_images = {}
        lq_subfolders = sorted(glob(os.path.join(self.lq_root, '*')))

        for lq_subfolder in lq_subfolders:
            subfolder_name = os.path.basename(lq_subfolder)
            lq_image_paths = sorted(list(scan_directory(lq_subfolder, full_path=True)))

            max_index = len(lq_image_paths)

            self.data_info["lq_path"].extend(lq_image_paths)
            self.data_info["folder"].extend([subfolder_name] * max_index)

            for index in range(max_index):
                self.data_info["index"].append(f"{index} / {max_index}")

            border_line = [0] * max_index

            for index in range(kwargs["num_frame"] // 2):
                border_line[index] = 1
                border_line[max_index - index - 1] = 1

            self.data_info["border"].extend(border_line)

            if self.cache_data:
                print(f"Cache {subfolder_name} for VideoTestDataset...")
                self.lq_images[subfolder_name] = read_image_sequence(lq_image_paths)
            else:
                self.lq_images[subfolder_name] = lq_image_paths

        self.folders = sorted(list(set(self.data_info["folder"])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            lq_images = self.lq_images[folder]
        else:
            lq_images = read_image_sequence(self.lq_images[folder])

        return {
            "low": lq_images,
            "folder": folder,
            "lq_path": self.lq_images[folder]
        }

    def __len__(self):
        return len(self.folders)


def modify_state_dict(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        origin_data: list = key.split('.')

        if origin_data[0] == "conv_first":
            origin_data[0] = "first_conv"
        elif origin_data[0] == "conv_after_body":
            origin_data[0] = "second_conv"
        elif origin_data[0] == "conv_before_upsample":
            origin_data[0] = "upsample_conv"
        elif origin_data[0] == "conv_last":
            origin_data[0] = "last_conv"
        elif origin_data[0] == "spynet":
            if origin_data[1] == "basic_module":
                origin_data[1] = "multi_modules"
                origin_data[3] = "module"
        elif origin_data[0][0:5] == "stage":
            if origin_data[0] == "stage8":
                if origin_data[1] == "0":
                    origin_data[0] = "pre_stage8"
                    origin_data[2] = str(int(origin_data[2]) - 1)
                    origin_data.pop(1)
                else:
                    origin_data[1] = str(int(origin_data[1]) - 1)
                index = 1
            else:
                index = 0

            if origin_data[1] == "residual_group1":
                origin_data[1] = "front_residual_group"
            elif origin_data[1] == "residual_group2":
                origin_data[1] = "back_residual_group"
            elif origin_data[1] == "pa_deform":
                origin_data[1] = "parallel_deform"
            elif origin_data[1] == "pa_fuse":
                origin_data[1] = "parallel_fusion"
                if origin_data[2] == "fc11":
                    origin_data[2] = "dual_fc"
                    origin_data.insert(3, "0")
                elif origin_data[2] == "fc12":
                    origin_data[2] = "dual_fc"
                    origin_data.insert(3, "1")
                elif origin_data[2] == "fc2":
                    origin_data[2] = "fc"
            elif origin_data[1] == "linear1":
                origin_data[1] = "front_linear"
            elif origin_data[1] == "linear2":
                origin_data[1] = "back_linear"
            elif origin_data[1] == "reshape":
                if origin_data[2] == "1":
                    origin_data[1] = "layer_norm"
                    origin_data.pop(2)
                elif origin_data[2] == "2":
                    origin_data[1] = "linear"
                    origin_data.pop(2)

            if len(origin_data) > 4 + index:
                if origin_data[4 + index] == "attn":
                    origin_data[4 + index] = "attention"
                    if origin_data[5 + index] == "qkv_self":
                        origin_data[5 + index] = "qkv"
                    elif origin_data[5 + index] == "qkv_mut":
                        origin_data[5 + index] = "qkv_mutual"
                    elif origin_data[5 + index] == "proj":
                        origin_data[5 + index] = "projection"
                elif origin_data[4 + index] == "norm1":
                    origin_data[4 + index] = "norm_list"
                    origin_data.insert(5 + index, "0")
                elif origin_data[4 + index] == "norm2":
                    origin_data[4 + index] = "norm_list"
                    origin_data.insert(5 + index, "1")
                elif origin_data[4 + index] == "mlp":
                    if origin_data[5 + index] == "fc11":
                        origin_data[5 + index] = "dual_fc"
                        origin_data.insert(6 + index, "0")
                    elif origin_data[5 + index] == "fc12":
                        origin_data[5 + index] = "dual_fc"
                        origin_data.insert(6 + index, "1")
                    elif origin_data[5 + index] == "fc2":
                        origin_data[5 + index] = "fc"

        new_key = '.'.join(origin_data)
        new_state_dict[new_key] = value

    return new_state_dict
