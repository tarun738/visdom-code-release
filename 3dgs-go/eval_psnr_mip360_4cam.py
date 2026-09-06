"""Aggregate PSNR over saved renders, averaged per scene then across scenes.

PSNR is computed from a single mean squared error pooled over all pixels and all three
colour channels, which is the definition used throughout the novel-view-synthesis
literature and the one our paper reports. `utils/image_utils.py` computes the same
quantity; `render.py` prints it per scene, and this script is for rolling several scenes
up into one number.
"""
import argparse
import glob
import os.path as osp

import numpy as np
from PIL import Image


def psnr(render_path, gt_path):
    render = np.asarray(Image.open(render_path), dtype=np.float64) / 255.0
    gt = np.asarray(Image.open(gt_path), dtype=np.float64) / 255.0
    mse = np.mean((render - gt) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", default="exps/gs_init",
                   help="directory holding <scene>/<cameras>/test/<checkpoint>/")
    p.add_argument("--scenes", nargs="+", required=True, help="scene names to average over")
    p.add_argument("--cameras", default="4", help="training-view count subdirectory")
    p.add_argument("--checkpoint", default="ours_best", help="checkpoint subdirectory")
    args = p.parse_args()

    scene_psnrs = []
    for scene in args.scenes:
        d = osp.join(args.base_dir, scene, args.cameras, "test", args.checkpoint)
        renders = sorted(glob.glob(osp.join(d, "renders", "*.png")))
        if not renders:
            print(f"{scene:14s} no renders under {d}")
            continue
        vals = [psnr(r, osp.join(d, "gt", osp.basename(r))) for r in renders]
        scene_psnrs.append(np.mean(vals))
        print(f"{scene:14s} PSNR: {scene_psnrs[-1]:.3f}  ({len(vals)} test images)")

    if scene_psnrs:
        print(f"\nAverage PSNR ({args.checkpoint}, {args.cameras} cameras, "
              f"{len(scene_psnrs)} scenes): {np.mean(scene_psnrs):.3f}")


if __name__ == "__main__":
    main()
