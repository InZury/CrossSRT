# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import cv2
import numpy as np


def compute_psnr(origin_image, target_image, border=0):
    if not origin_image.shape == target_image.shape:
        raise ValueError("Input images must have the same dimensions!")

    height, width = origin_image.shape[:2]
    origin_image = origin_image[border:height - border, border:width - border]
    origin_image = origin_image.astype(np.float64)
    target_image = target_image[border:height - border, border:width - border]
    target_image = target_image.astype(np.float64)

    mse = np.mean((origin_image - target_image) ** 2)  # Mean Squared Error

    if mse == 0:
        return float("inf")

    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(origin_image, target_image, border=0):
    def ssim(_origin_image, _target_image):
        constant_1 = (0.01 * 255.0) ** 2
        constant_2 = (0.03 * 255.0) ** 2

        _origin_image = _origin_image.astype(np.float64)
        _target_image = _target_image.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        origin_mean = cv2.filter2D(_origin_image, -1, window)[5:-5, 5:-5]  # Symbol mu
        target_mean = cv2.filter2D(_target_image, -1, window)[5:-5, 5:-5]
        squared_origin_mean = origin_mean ** 2
        squared_target_mean = target_mean ** 2
        multiple_mean = origin_mean * target_mean
        origin_sigma = cv2.filter2D(_origin_image ** 2, -1, window)[5:-5, 5:-5] - squared_origin_mean
        target_sigma = cv2.filter2D(_target_image ** 2, -1, window)[5:-5, 5:-5] - squared_target_mean
        multiple_sigma = cv2.filter2D(_origin_image * _target_image, -1, window)[5:-5, 5:-5] - multiple_mean

        ssim_map = (
            ((2 * multiple_mean + constant_1) * (2 * multiple_sigma + constant_2)) /
            ((squared_origin_mean + squared_target_mean + constant_1) * (origin_sigma + target_sigma + constant_2))
        )

        return ssim_map.mean()

    if not origin_image.shape == target_image.shape:
        raise ValueError("Input images must have the same dimensions!")

    height, width = origin_image.shape[:2]
    origin_image = origin_image[border:height - border, border:width - border]
    target_image = target_image[border:height - border, border:width - border]

    if origin_image.ndim == 2:
        return ssim(origin_image, target_image)
    elif origin_image.ndim == 3:
        if origin_image.shape[2] == 3:
            ssim_list = []

            for index in range(3):
                ssim_list.append(ssim(origin_image[:, :, index], target_image[:, :, index]))

            return np.array(ssim_list).mean()
        elif origin_image.shape[2] == 1:
            return ssim(np.squeeze(origin_image), np.squeeze(target_image))
    else:
        raise ValueError("Wrong input image dimensions!")

