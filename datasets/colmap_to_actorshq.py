import argparse
import collections
import os
import struct

import cv2
import numpy as np
import csv
from collections import OrderedDict
from scipy.spatial.transform import Rotation

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = OrderedDict()
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_name] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def writeColmapCameras(outsubdir, cam_extrinsics, cam_intrinsics, image_dims=None, image_fnames=None):
    out_calib_file = os.path.join(outsubdir, 'calibration.csv')
    os.makedirs(outsubdir, exist_ok=True)

    with open(out_calib_file, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        header = ['name', 'w', 'h', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'fx', 'fy', 'px', 'py']
        writer.writerow(header)

        iter_list = image_fnames if image_fnames is not None else cam_extrinsics.keys()
        for i, key in enumerate(iter_list):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]

            imageH = intr.height
            imageW = intr.width
            fx = intr.params[0] / imageW
            fy = intr.params[1] / imageH
            cx = intr.params[2] / imageW
            cy = intr.params[3] / imageH

            if image_dims is not None:
                imageH, imageW = image_dims

            R = qvec2rotmat(extr.qvec)
            rvec = Rotation.from_matrix(R.T).as_rotvec()
            tvec = -np.array(extr.tvec) @ R

            row = ['Cam{0:03d}'.format(i + 1), imageW, imageH, rvec[0], rvec[1], rvec[2],
                   tvec[0], tvec[1], tvec[2], fx, fy, cx, cy]
            writer.writerow(row)
    return image_fnames


def writeImages(outsubdir, datadir, image_fnames, image_subdir, mask_subdir, timestamp, cam_extrinsics=None):
    img_outdir = os.path.join(outsubdir, 'rgbs')
    mask_outdir = os.path.join(outsubdir, 'masks')
    os.makedirs(img_outdir, exist_ok=True)
    os.makedirs(mask_outdir, exist_ok=True)

    image_dir = os.path.join(datadir, image_subdir)
    mask_dir = os.path.join(datadir, mask_subdir)
    img_size = None

    iter_list = sorted(cam_extrinsics.keys()) if cam_extrinsics is not None else image_fnames
    for i, fname in enumerate(iter_list):
        img_fname = os.path.join(image_dir, fname)
        mask_fname = os.path.join(mask_dir, fname[:-3] + 'png')

        img = cv2.imread(img_fname, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_fname, cv2.IMREAD_GRAYSCALE)

        if img_size is None:
            img_size = img.shape[:2]

        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img_subsubdir = os.path.join(img_outdir, 'Cam{0:03d}'.format(i + 1))
        mask_subsubdir = os.path.join(mask_outdir, 'Cam{0:03d}'.format(i + 1))
        os.makedirs(img_subsubdir, exist_ok=True)
        os.makedirs(mask_subsubdir, exist_ok=True)

        cv2.imwrite(os.path.join(img_subsubdir, 'Cam{0:03d}_rgb{1:06d}.jpg'.format(i + 1, timestamp)), img)
        cv2.imwrite(os.path.join(mask_subsubdir, 'Cam{0:03d}_mask{1:06d}.png'.format(i + 1, timestamp)), resized_mask)

    return img_size


def convert(colmap_dir, outdir, image_subdir='images_2', mask_subdir='masks', timestamp=0):
    images = read_extrinsics_binary(os.path.join(colmap_dir, "sparse/0", "images.bin"))
    cameras = read_intrinsics_binary(os.path.join(colmap_dir, "sparse/0", "cameras.bin"))

    image_fnames = sorted(images.keys())

    img_size = writeImages(outdir, colmap_dir, image_fnames, image_subdir, mask_subdir, timestamp, cam_extrinsics=images)
    writeColmapCameras(outdir, images, cameras, image_dims=img_size, image_fnames=image_fnames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a COLMAP sparse model (e.g. MipNeRF360, OmniObject3D) into "
                    "the ActorsHQ calibration.csv / rgbs / masks layout used by zipnerf.")
    parser.add_argument("colmap_dir", help="Dataset root containing sparse/0/{images,cameras}.bin, "
                                            "the image_subdir, and masks/.")
    parser.add_argument("outdir", help="Output directory for calibration.csv, rgbs/, masks/.")
    parser.add_argument("--image_subdir", default="images_2",
                        help="Subdirectory of colmap_dir with the RGB images referenced by images.bin (default: images_2).")
    parser.add_argument("--mask_subdir", default="masks",
                        help="Subdirectory of colmap_dir with per-image masks, same filenames as images (default: masks).")
    parser.add_argument("--timestamp", type=int, default=0,
                        help="Frame index to embed in the ActorsHQ output filenames (Cam001_rgb000000.jpg). "
                            "These datasets are single-frame, so this is just a naming placeholder (default: 0).")
    opt = parser.parse_args()

    if not os.path.exists(opt.colmap_dir):
        raise FileNotFoundError(f"colmap_dir does not exist: {opt.colmap_dir}")

    convert(opt.colmap_dir, opt.outdir, opt.image_subdir, opt.mask_subdir, opt.timestamp)
    print(f"Wrote ActorsHQ-format dataset to {opt.outdir}")
