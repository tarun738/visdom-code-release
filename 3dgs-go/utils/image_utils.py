#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from skimage.metrics import structural_similarity

def _as_batch(img):
    """Treat an unbatched [C, H, W] image as a batch of one.

    Both metrics below reduce over everything after the leading dimension, which is the
    intended per-image reduction only when that dimension indexes images. Handed a bare
    [C, H, W] tensor it indexes colour channels instead, giving one value per channel;
    averaging those per-channel PSNRs is the geometric rather than the arithmetic mean of
    the channel MSEs, which overstates PSNR whenever the channels differ.
    """
    return img.unsqueeze(0) if img.dim() == 3 else img

def mse(img1, img2):
    img1, img2 = _as_batch(img1), _as_batch(img2)
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))

def load_meshlab_file(meshlab_file):
    ### in default, the meshlab file named pcd_transform.txt
    with open(meshlab_file) as f:
        for num, line in enumerate(f, 1):
            if num == 6:
                MLMatrix44 = line
                break
    return np.array(MLMatrix44.split()).reshape(4, 4).astype(np.float32)

def _to_hwc(img):
    """Torch [C, H, W] or [N, C, H, W] -> numpy [H, W, C]; a batch is averaged by caller."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError("ssim_metric expects a single image, got a batch of "
                             f"{img.shape[0]}")
        img = img[0]
    return np.transpose(img, (1, 2, 0))


def ssim_metric(img1, img2):
    """SSIM as reported for the paper: scikit-image's default window, over colour
    channels, on floats in [0, 1].

    Deliberately separate from utils.loss_utils.ssim. That one is the D-SSIM term in the
    training objective and uses an 11x11 Gaussian window; changing it would change what
    the model optimises. This function only reports.
    """
    return float(structural_similarity(_to_hwc(img1), _to_hwc(img2),
                                       channel_axis=2, data_range=1.0))
