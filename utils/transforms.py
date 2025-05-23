import cv2
import torch
import numpy as np


def image2tensor(images, bgr2rgb=True, is_float=True):
    def _to_tensor(image, _bgr2rgb, _is_float):
        if image.shape[2] == 3 and _bgr2rgb:
            if image.dtype == "float64":
                image = image.astype("float32")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image.transpose(2, 0, 1))

        if _is_float:
            image = image.float()

        return image

    if isinstance(images, list):
        return [_to_tensor(image, bgr2rgb, is_float) for image in images]
    else:
        return _to_tensor(images, bgr2rgb, is_float)


def rgb2ycc(image, only_y=True):
    # convert color channel from rgb to ycc(yCbCr)

    image_type = image.dtype
    image.astype(np.float32)

    if image_type != np.uint8:
        image *= 255.0

    if only_y:
        result = np.dot(image, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        transform_matrix = [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]
        result = image @ transform_matrix / 255.0 + np.array([16.0, 128.0, 128.0])

    if image_type == np.uint8:
        result = result.round()
    else:
        result /= 255.0

    return result.astype(image_type)


def ycc2rgb(image):
    # convert color channel from ycc to rgb

    image_type = image.dtype
    image.astype(np.float32)

    if image_type != np.uint8:
        image *= 255.0

    transform_matrix = np.array(
        [[0.456621, 0.456621, 0.456621], [0.0, -0.153632, 0.791071], [0.625893, -0.318811, 0.0]]
    ) * 0.01
    result = image @ transform_matrix * 255.0 + np.array([-222.921, 135.576, -276.836])
    result = np.clip(result, 0, 255)

    if image_type == np.uint8:
        result = result.round()
    else:
        result /= 255.0

    return result.astype(image_type)


def bgr2ycc(image, only_y=True):
    # convert color channel from bgr to ycc(yCbCr)

    image_type = image.dtype
    image.astype(np.float32)

    if image_type != np.uint8:
        image *= 255.0

    if only_y:
        result = np.dot(image, [24.966, 128.533, 65.481]) / 255.0 + 16.0
    else:
        transform_matrix = [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]
        result = image @ transform_matrix / 255.0 + np.array([16.0, 128.0, 128.0])

    if image_type == np.uint8:
        result = result.round()
    else:
        result /= 255.0

    return result.astype(image_type)


def mod_crop(input_image, scale):
    image = np.copy(input_image)

    if image.ndim == 2:
        height, width = image.shape
        resolution = [height % scale, width % scale]
        image = image[:height - resolution[0], :width - resolution[1]]
    elif image.ndim == 3:
        height, width, channel = image.shape
        resolution = [height % scale, width % scale]
        image = image[:height - resolution[0], :width - resolution[1], :]
    else:
        raise ValueError(f"Wrong image ndim: [{image.ndim:d}]")

    return image
