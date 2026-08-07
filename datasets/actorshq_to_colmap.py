import argparse
import os

import cv2
from scipy.spatial.transform import Rotation

from camera_actorshq import read_calibration_csv


def prepare_train_data_colmap(datapath, outdir, timestamp=0):
    """Read an ActorsHQ calibration.csv + rgbs/masks and write a COLMAP dataset
    (images/, masks/, sparse/0/{cameras,images,points3D}.txt) into outdir, ready
    for our 3dgs-go pipeline.

    Camera poses/intrinsics are known exactly from the ActorsHQ calibration, so
    we write them directly as the COLMAP sparse model instead of running COLMAP's
    own feature matching + triangulation: point_triangulator would only refine
    points3D (unused, since gaussians are initialized from a visual hull instead,
    see visual_hull.py) and leaves the input camera poses untouched."""
    calib = os.path.join(datapath, 'calibration.csv')
    cams = read_calibration_csv(calib)

    num_cameras = len(cams)
    print("Cameras: {}".format(num_cameras))

    image_dir = os.path.join(datapath, 'rgbs')
    mask_dir = os.path.join(datapath, 'masks')
    outdir_img = os.path.join(outdir, 'images')
    outdir_mask = os.path.join(outdir, 'masks')
    outdir_sparse = os.path.join(outdir, 'sparse', '0')

    os.makedirs(outdir_img, exist_ok=True)
    os.makedirs(outdir_mask, exist_ok=True)
    os.makedirs(outdir_sparse, exist_ok=True)

    print("Copying images to {}".format(outdir_img))
    for subdir in sorted(os.listdir(image_dir)):
        for file in sorted(os.listdir(os.path.join(image_dir, subdir))):
            if 'rgb{0:06d}'.format(timestamp) not in file or not file.endswith('jpg'):
                continue
            filename = os.path.join(image_dir, subdir, file)
            mask_filename = os.path.join(mask_dir, subdir, file.replace('rgb', 'mask')[:-4] + '.png')

            filenew = os.path.join(outdir_img, file)
            mask_filenew = os.path.join(outdir_mask, file.replace('mask', 'rgb')[:-4] + '.png')

            img = cv2.imread(filename)
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            img[mask == 0] = 255  # white background
            cv2.imwrite(filenew, img)
            cv2.imwrite(mask_filenew, mask)

    # CAMERA_ID, MODEL, WIDTH, HEIGHT, FX, FY, CX, CY
    print("Writing sparse/0/{{cameras,images,points3D}}.txt to {}".format(outdir_sparse))
    with open(os.path.join(outdir_sparse, 'cameras.txt'), 'w') as fout:
        for i in range(num_cameras):
            fx = cams[i].fx_pixel
            fy = cams[i].fy_pixel
            cx = cams[i].cx_pixel
            cy = cams[i].cy_pixel
            width = cams[i].width
            height = cams[i].height
            fout.write("{0} PINHOLE {1:d} {2:d} {3:0.6f} {4:0.6f} {5:0.6f} {6:0.6f}\n".format(
                i + 1, width, height, fx, fy, cx, cy))

    with open(os.path.join(outdir_sparse, 'images.txt'), 'w') as fout:
        for i in range(num_cameras):
            R = Rotation.from_rotvec(-cams[i].rotation_axisangle)
            quat = R.as_quat()
            quat = quat[[3, 0, 1, 2]]
            transl = -R.as_matrix() @ cams[i].translation
            fout.write("{0} {1:0.06f} {2:0.06f} {3:0.06f} {4:0.06f} {5:0.06f} {6:0.06f} {7:0.06f} {8} Cam{9:03d}_rgb{10:06d}.jpg\n\n".format(
                i + 1, quat[0], quat[1], quat[2], quat[3], transl[0], transl[1], transl[2], i + 1, i + 1, timestamp))

    with open(os.path.join(outdir_sparse, 'points3D.txt'), 'w') as fout:
        fout.write("# Empty file\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an ActorsHQ-format sequence (calibration.csv + rgbs/ + masks/) into a "
                    "COLMAP dataset for our 3dgs-go pipeline.")
    parser.add_argument("--data", type=str, required=True, help="Path to the ActorsHQ sequence root.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for the COLMAP dataset.")
    parser.add_argument("--timestamp", type=int, default=0,
                        help="Frame index to extract from the ActorsHQ sequence (default: 0).")
    opt = parser.parse_args()

    prepare_train_data_colmap(opt.data, opt.outdir, opt.timestamp)

    print("Done")
