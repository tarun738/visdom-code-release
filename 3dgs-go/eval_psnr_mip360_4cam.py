import glob
import os.path as osp
import sys

import numpy as np
from PIL import Image

BASE_DIR = "exps/gs_init"
SCENES = ["bonsai", "kitchen", "garden"]
CAM = "4"
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "ours_best"


def psnr_correct(render_path, gt_path):
    """PSNR from a single combined MSE over all pixels and channels (not an
    average of per-channel PSNRs, which is what utils/image_utils.py computes
    for unbatched [C,H,W] tensors -- see the earlier discussion)."""
    render = np.asarray(Image.open(render_path), dtype=np.float64) / 255.0
    gt = np.asarray(Image.open(gt_path), dtype=np.float64) / 255.0
    mse = np.mean((render - gt) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))


scene_psnrs = []
for scene in SCENES:
    checkpoint_dir = osp.join(BASE_DIR, scene, CAM, "test", CHECKPOINT)
    render_paths = sorted(glob.glob(osp.join(checkpoint_dir, "renders", "*.png")))

    image_psnrs = []
    for render_path in render_paths:
        gt_path = osp.join(checkpoint_dir, "gt", osp.basename(render_path))
        image_psnrs.append(psnr_correct(render_path, gt_path))

    scene_psnr = sum(image_psnrs) / len(image_psnrs)
    scene_psnrs.append(scene_psnr)
    print(f"{scene:10s} PSNR: {scene_psnr:.3f}  ({len(image_psnrs)} test images)")

print(f"\nAverage PSNR ({CHECKPOINT}, {CAM} cameras, {len(SCENES)} scenes): {sum(scene_psnrs) / len(scene_psnrs):.3f}")
