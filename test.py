# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import os
import torch
import numpy as np

from sys import stderr
from omegaconf import OmegaConf
from collections import OrderedDict
from torch.utils.data import DataLoader

from models.csrt import CSRT
from utils.datasets import VideoRecurrentTestDataset, SingleVideoRecurrentTestDataset, modify_state_dict
from utils.video import test_video
from utils.eval_matric import compute_psnr, compute_ssim
from utils.transforms import bgr2ycc


if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Super_Resolution_REDS_6_Frames")
    parser.add_argument("--config", type=str, default="csrt_base.yaml")
    parser.add_argument("--noise", type=int, default=0, help="Noise level for denoising, 10, 20, 30, 40, 50")
    parser.add_argument("--lq_data", type=str, default="data/REDS4/lq")
    parser.add_argument("--gt_data", type=str, default="None")
    parser.add_argument("--tile", type=int, nargs='+', default=[40, 128, 128])
    parser.add_argument("--tile_overlap", type=int, nargs='+', default=[2, 20, 20])
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # Environment setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.join("config", "models"), args.config)
    config = OmegaConf.load(config_path)

    # Set Model
    model = CSRT(config)
    args.scale = config.parameters.upscale
    args.window_size = config.parameters.window_size
    args.non_blind_denoising = config.parameters.non_blind_denoising

    model_path = f"{os.path.join(os.path.join('models', 'model_zoo'), args.task)}.pth"
    pretrained_model = torch.load(model_path)

    model.load_state_dict(modify_state_dict(pretrained_model["params"]), strict=True)

    # Test model
    model.eval()
    model = model.to(device)

    data_config = {
        "gt_data": args.gt_data,
        "lq_data": args.lq_data,
        "sigma": args.noise,
        "num_frame": -1,
        "cache_data": False
    }

    if args.gt_data is not None:
        test_dataset = VideoRecurrentTestDataset(data_config)
    else:
        test_dataset = SingleVideoRecurrentTestDataset(data_config)

    test_loader = DataLoader(
        dataset=test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False
    )

    save_directory = os.path.join("output", args.task)

    if args.save:
        os.makedirs(save_directory, exist_ok=True)

    outputs = OrderedDict()
    outputs["psnr"] = []
    outputs["ssim"] = []
    outputs["psnr_y"] = []
    outputs["ssim_y"] = []

    assert len(test_loader) != 0, f"No dataset found at {args.lq_data}!"

    global_gt = None

    print("Test Model", file=stderr)

    for indices, batch in enumerate(test_loader):
        if not isinstance(batch, dict):
            raise TypeError("batch type must be a dict!")

        lq = batch["low"].to(device)
        folder = batch["folder"]
        gt = global_gt = batch["high"] if "high" in batch else None

        with torch.no_grad():
            output = test_video(lq, model, args)

        outputs_folder = OrderedDict()
        outputs_folder["psnr"] = []
        outputs_folder["ssim"] = []
        outputs_folder["psnr_y"] = []
        outputs_folder["ssim_y"] = []

        for index in range(output.shape[1]):
            image = output[:, index, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()

            if image.ndim == 3:
                image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))

            image = (image * 255.0).round().astype(np.uint8)

            if args.save:
                sequence = f"{os.path.basename(batch['lq_path'][index][0]).split('.')[0]}.png"
                os.makedirs(os.path.join(save_directory, folder[0]), exist_ok=True)
                cv2.imwrite(os.path.join(os.path.join(save_directory, folder[0]), sequence), image)

            if gt is not None:
                gt_image = gt[:, index, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()

                if gt_image.ndim == 3:
                    gt_image = np.transpose(gt_image[[2, 1, 0], :, :], (1, 2, 0))

                gt_image = (gt_image * 255.0).round().astype(np.uint8)
                gt_image = np.squeeze(gt_image)

                outputs_folder["psnr"].append(compute_psnr(gt_image, image, border=0))
                outputs_folder["ssim"].append(compute_ssim(gt_image, image, border=0))

                if gt_image.ndim == 3:
                    image = bgr2ycc(image.astype(np.float32) / 255.0) * 255.0
                    gt_image = bgr2ycc(gt_image.astype(np.float32) / 255.0) * 255.0
                    outputs_folder["psnr_y"].append(compute_psnr(gt_image, image))
                    outputs_folder["ssim_y"].append(compute_ssim(gt_image, image))
                else:
                    outputs_folder["psnr_y"] = outputs_folder["psnr"]
                    outputs_folder["ssim_y"] = outputs_folder["ssim"]

        if gt is not None:
            psnr = sum(outputs_folder["psnr"]) / len(outputs_folder["psnr"])
            ssim = sum(outputs_folder["ssim"]) / len(outputs_folder["ssim"])
            psnr_y = sum(outputs_folder["psnr_y"]) / len(outputs_folder["psnr_y"])
            ssim_y = sum(outputs_folder["ssim_y"]) / len(outputs_folder["ssim_y"])

            outputs["psnr"].append(psnr)
            outputs["ssim"].append(ssim)
            outputs["psnr_y"].append(psnr_y)
            outputs["ssim_y"].append(ssim_y)

            print(
                f"Testing {folder[0]:20s} ({indices+1:2d}/{len(test_loader)}) "
                f"- PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | PSNR_Y: {psnr_y:.2f} dB | SSIM_Y: {ssim_y:.4f}"
            )
        else:
            print(f"Testing {folder[0]:20s} ({indices:2d}/{len(test_loader)})")

    if global_gt is not None:
        average_psnr = sum(outputs["psnr"]) / len(outputs["psnr"])
        average_ssim = sum(outputs["ssim"]) / len(outputs["ssim"])
        average_psnr_y = sum(outputs["psnr_y"]) / len(outputs["psnr_y"])
        average_ssim_y = sum(outputs["ssim_y"]) / len(outputs["ssim_y"])

        print(
            f"\n"
            f"{save_directory}-- "
            f"Average PSNR: {average_psnr:.2f} dB | SSIM: {average_ssim:.4f} | "
            f"PSNR_Y: {average_psnr_y:.2f} dB | SSIM_Y: {average_ssim_y:.4f}"
        )
