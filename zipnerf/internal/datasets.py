import abc
import copy
import json
import os
from pathlib import Path
import cv2
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import raw_utils
from internal import utils
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
# This is ugly, but it works.
import sys
import csv
sys.path.insert(0, 'internal/pycolmap')
sys.path.insert(0, 'internal/pycolmap/pycolmap')
import pycolmap
import skimage
import trimesh
from internal import compnormals

def load_dataset(split, train_dir, config: configs.Config):
    """Loads a split of a dataset using the data_loader specified by `config`."""
    if config.multiscale:
        dataset_dict = {
            'llff': MultiLLFF,
        }
    else:
        dataset_dict = {
            'actorshq': ActorsHQ,
            'blender': Blender,
            'llff': LLFF,
            'tat_nerfpp': TanksAndTemplesNerfPP,
            'tat_fvs': TanksAndTemplesFVS,
            'dtu': DTU,
        }
    return dataset_dict[config.dataset_loader](split, train_dir, config)


class NeRFSceneManager(pycolmap.SceneManager):
    """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

    def process(self):
        """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

        self.load_cameras()
        self.load_images()
        # self.load_points3D()  # For now, we do not need the point cloud data.

        # Assume shared intrinsics between all cameras.
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == 'PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == 'RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == 'OPENCV':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['p1'] = cam.p1
            params['p2'] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['k3'] = cam.k3
            params['k4'] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype


def load_blender_posedata(data_dir, split=None):
    """Load poses from `transforms.json` file, as used in Blender/NGP datasets."""
    suffix = '' if split is None else f'_{split}'
    pose_file = os.path.join(data_dir, f'transforms{suffix}.json')
    with utils.open_file(pose_file, 'r') as fp:
        meta = json.load(fp)
    names = []
    poses = []
    for _, frame in enumerate(meta['frames']):
        filepath = os.path.join(data_dir, frame['file_path'])
        if utils.file_exists(filepath):
            names.append(frame['file_path'].split('/')[-1])
            poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
    poses = np.stack(poses, axis=0)

    w = meta['w']
    h = meta['h']
    cx = meta['cx'] if 'cx' in meta else w / 2.
    cy = meta['cy'] if 'cy' in meta else h / 2.
    if 'fl_x' in meta:
        fx = meta['fl_x']
    else:
        fx = 0.5 * w / np.tan(0.5 * float(meta['camera_angle_x']))
    if 'fl_y' in meta:
        fy = meta['fl_y']
    else:
        fy = 0.5 * h / np.tan(0.5 * float(meta['camera_angle_y']))
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))
    coeffs = ['k1', 'k2', 'p1', 'p2']
    if not any([c in meta for c in coeffs]):
        params = None
    else:
        params = {c: (meta[c] if c in meta else 0.) for c in coeffs}
    camtype = camera_utils.ProjectionType.PERSPECTIVE
    return names, poses, pixtocam, params, camtype


class Dataset(torch.utils.data.Dataset):
    """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    exposures: optional per-image exposure value (shutter * ISO / 1000).
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    metadata: dict, optional metadata for raw datasets.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_exposures: optional list of exposure values for the render path.
    render_path: bool, indicates if a smooth camera path should be generated.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

    def __init__(self,
                 split: str,
                 data_dir: str,
                 config: configs.Config):
        super().__init__()

        # Initialize attributes
        self._patch_size = max(config.patch_size, 1)
        self._batch_size = config.batch_size // config.world_size
        if self._patch_size ** 2 > self._batch_size:
            raise ValueError(f'Patch size {self._patch_size}^2 too large for ' +
                             f'per-process batch size {self._batch_size}')
        self._batching = utils.BatchingMethod(config.batching)
        self._use_tiffs = config.use_tiffs
        self._load_disps = config.compute_disp_metrics
        self._load_normals = config.compute_normal_metrics
        self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
        self._apply_bayer_mask = config.apply_bayer_mask
        self._render_spherical = False

        self.config = config
        self.global_rank = config.global_rank
        self.world_size = config.world_size
        self.split = utils.DataSplit(split)
        self.data_dir = data_dir
        self.near = config.near
        self.far = config.far
        self.render_path = config.render_path
        self.distortion_params = None
        self.disp_images = None
        self.normal_images = None
        self.alphas = None
        self.poses = None
        self.pixtocam_ndc = None
        self.metadata = None
        self.camtype = camera_utils.ProjectionType.PERSPECTIVE
        self.exposures = None
        self.render_exposures = None

        # Providing type comments for these attributes, they must be correctly
        # initialized by _load_renderings() (see docstring) in any subclass.
        self.images: np.ndarray = None
        self.camtoworlds: np.ndarray = None
        self.pixtocams: np.ndarray = None
        self.height: int = None
        self.width: int = None

        # Load data from disk using provided config parameters.
        self._load_renderings(config)

        if self.render_path and (not config.dataset_loader == 'actorshq'):
            if config.render_path_file is not None:
                with utils.open_file(config.render_path_file, 'rb') as fp:
                    render_poses = np.load(fp)
                self.camtoworlds = render_poses
            if config.render_resolution is not None:
                self.width, self.height = config.render_resolution
            if config.render_focal is not None:
                self.focal = config.render_focal
            if config.render_camtype is not None:
                if config.render_camtype == 'pano':
                    self._render_spherical = True
                else:
                    self.camtype = camera_utils.ProjectionType(config.render_camtype)

            self.distortion_params = None
            self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                       self.height)

        self._n_examples = self.camtoworlds.shape[0]

        self.cameras = (self.pixtocams,
                        self.camtoworlds,
                        self.distortion_params,
                        self.pixtocam_ndc)

        # Seed the queue with one batch to avoid race condition.
        if self.split == utils.DataSplit.TRAIN and not config.compute_visibility:
            self._next_fn = self._next_train
        else:
            self._next_fn = self._next_test

        # self._next_train(0)
        # self.generate_ray_batch(0)
        # if not self.split == utils.DataSplit.TRAIN:
        #     import pdb; pdb.set_trace()
        #     restest = [self._next_fn(i) for i in range(self._n_examples)]

    @property
    def size(self):
        return self._n_examples

    def __len__(self):
        if self.split == utils.DataSplit.TRAIN and not self.config.compute_visibility:
            return 1000
        else:
            return self._n_examples

    @abc.abstractmethod
    def _load_renderings(self, config):
        """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters.
    In inherited classes, this method must set the following public attributes:
      images: [N, height, width, 3] array for RGB images.
      disp_images: [N, height, width] array for depth data (optional).
      normal_images: [N, height, width, 3] array for normals (optional).
      camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
      poses: [..., 3, 4] array of auxiliary pose data (optional).
      pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
      distortion_params: dict, camera lens distortion model parameters.
      height: int, height of images.
      width: int, width of images.
      focal: float, focal length to use for ideal pinhole rendering.
    """

    def _make_ray_batch(self,
                        pix_x_int,
                        pix_y_int,
                        cam_idx,
                        lossmult=None
                        ):
        """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """
        broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
        ray_kwargs = {
            'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
            'near': broadcast_scalar(self.near),
            'far': broadcast_scalar(self.far),
            'cam_idx': broadcast_scalar(cam_idx),
        }
        # Collect per-camera information needed for each ray.
        if self.metadata is not None:
            # Exposure index and relative shutter speed, needed for RawNeRF.
            for key in ['exposure_idx', 'exposure_values']:
                idx = 0 if self.render_path else cam_idx
                ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
        if self.exposures is not None:
            idx = 0 if self.render_path else cam_idx
            ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
        if self.render_path and self.render_exposures is not None:
            ray_kwargs['exposure_values'] = broadcast_scalar(
                self.render_exposures[cam_idx])

        pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

        # Slow path, do ray computation using numpy (on CPU).
        batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)
        batch['cam_dirs'] = -self.camtoworlds[ray_kwargs['cam_idx'][..., 0]][..., :3, 2]

        # import trimesh
        # pts = batch['origins'][..., None, :] + batch['directions'][..., None, :] * np.linspace(0, 1, 5)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test.ply", "ply")
        #
        # pts = batch['origins'][0, 0, None, :] - self.camtoworlds[cam_idx][:, 2] * np.linspace(0, 1, 100)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test2.ply", "ply")

        if not self.render_path:
            print(cam_idx)
            print(pix_y_int)
            print(self.images[cam_idx].shape)
            print(pix_y_int)
            batch['rgb'] = self.images[cam_idx][pix_y_int, pix_x_int]
        if self._load_disps:
            batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
        if self._load_normals:
            batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
            batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_train(self, item):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = self._batch_size // self._patch_size ** 2
        lower_border = self._num_border_pixels_to_mask
        upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
        # Random pixel patch x-coordinates.
        pix_x_int = np.random.randint(lower_border, self.width - upper_border,
                                      (num_patches, 1, 1))
        # Random pixel patch y-coordinates.
        pix_y_int = np.random.randint(lower_border, self.height - upper_border,
                                      (num_patches, 1, 1))
        # Add patch coordinate offsets.
        # Shape will broadcast to (num_patches, _patch_size, _patch_size).
        patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
            self._patch_size, self._patch_size)
        pix_x_int = pix_x_int + patch_dx_int
        pix_y_int = pix_y_int + patch_dy_int
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
        else:
            cam_idx = np.random.randint(0, self._n_examples, (1,))

        if self._apply_bayer_mask:
            # Compute the Bayer mosaic mask for each pixel in the batch.
            lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
        else:
            lossmult = None

        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                    lossmult=lossmult)

    def generate_ray_batch(self, cam_idx: int):
        """Generate ray batch for a specified camera in the dataset."""
        if self._render_spherical:
            camtoworld = self.camtoworlds[cam_idx]
            rays = camera_utils.cast_spherical_rays(
                camtoworld, self.height, self.width, self.near, self.far)
            return rays
        else:
            # Generate rays for all pixels in the image.
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                self.width, self.height)
            return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

    def _next_test(self, item):
        """Sample next test batch (one full image)."""
        return self.generate_ray_batch(item)

    def collate_fn(self, item):
        return self._next_fn(item[0])

    def __getitem__(self, item):
        return self._next_fn(item)


class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the blender dataset.')
        pose_file = os.path.join(self.data_dir, f'transforms_{self.split.value}.json')
        with utils.open_file(pose_file, 'r') as fp:
            meta = json.load(fp)
        images = []
        disp_images = []
        normal_images = []
        cams = []
        for idx, frame in enumerate(tqdm(meta['frames'], desc='Loading Blender dataset', disable=self.global_rank != 0, leave=False)):
            fprefix = os.path.join(self.data_dir, frame['file_path'])

            def get_img(f, fprefix=fprefix):
                image = utils.load_img(fprefix + f)
                if config.factor > 1:
                    image = lib_image.downsample(image, config.factor)
                return image

            if self._use_tiffs:
                channels = [get_img(f'_{ch}.tiff') for ch in ['R', 'G', 'B', 'A']]
                # Convert image to sRGB color space.
                image = lib_image.linear_to_srgb_np(np.stack(channels, axis=-1))
            else:
                image = get_img('.png') / 255.
            images.append(image)

            if self._load_disps:
                disp_image = get_img('_disp.tiff')
                disp_images.append(disp_image)
            if self._load_normals:
                normal_image = get_img('_normal.png')[..., :3] * 2. / 255. - 1.
                normal_images.append(normal_image)

            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))

        self.images = np.stack(images, axis=0)
        if self._load_disps:
            self.disp_images = np.stack(disp_images, axis=0)
        if self._load_normals:
            self.normal_images = np.stack(normal_images, axis=0)
            self.alphas = self.images[..., -1]

        rgb, alpha = self.images[..., :3], self.images[..., -1:]
        self.images = rgb * alpha + (1. - alpha)  # Use a white background.
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = np.stack(cams, axis=0)
        self.focal = .5 * self.width / np.tan(.5 * float(meta['camera_angle_x']))
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)
def load_actorshq(path, cameraIds, timestamp, scalefactor = 1.0):
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.

    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics.
    """
    from scipy.spatial.transform import Rotation
    poses_all = {}
    intrinsics_all = {}
    image_dims = {}
    names = {}
    with open(os.path.join(path,'calibration.csv'), "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            camId = int(row["name"][-3:])
            tfm_cam2world = np.eye(4)
            tfm_cam2world[:3, :3] = Rotation.from_rotvec(np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])])).as_matrix()
            tfm_cam2world[:3, 3] = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            poses_all[camId] = tfm_cam2world
            intrinsics_all[camId] = np.array([[int(row['w'])*float(row["fx"]), 0, int(row['w'])*float(row["px"])],[0, int(row['h'])*float(row["fy"]), int(row['h'])*float(row["py"])],[0, 0, 1]])
            image_dims[camId] = [int(row['w']), int(row['h'])]
            names[camId] = row["name"]

    allposes = []
    for k in poses_all:
        allposes.append(poses_all[k])
    allposes = np.stack(allposes,axis=0)

    allintrinsics = []
    for k in intrinsics_all:
        allintrinsics.append(intrinsics_all[k])
    allintrinsics = np.stack(allintrinsics,axis=0)


    allimgdims = []
    for k in image_dims:
        allimgdims.append(image_dims[k])
    allimgdims =  np.stack(allimgdims,axis=0)

    allnames =[]
    for k in names:
        allnames.append(names[k])
    allnames = np.stack(allnames,axis=0)
    print(cameraIds)
    names = [names[i] for i in cameraIds]
    poses_all = np.stack([poses_all[i] for i in cameraIds],axis=0) # [C, 4, 4]
    intrinsics_all =  np.stack([intrinsics_all[i] for i in cameraIds],axis=0) # [C, 3, 3]
    image_dims = np.stack([image_dims[i] for i in cameraIds],axis=0)
    center = np.mean(allposes[:,:3,-1],axis=0)
    poses_all[:,:3,-1] -= center
    allposes[:,:3,-1] -= center
    scale = np.max(np.linalg.norm(allposes[:,:3,-1],axis=-1))
    #TODO
    # poses_all[:,:3,-1] += center
    # allposes[:,:3,-1] += center

    poses_all[:,:3,-1] /= scale / scalefactor
    allposes[:,:3,-1] /= scale / scalefactor

    return poses_all, intrinsics_all, image_dims, allposes, allintrinsics, allimgdims, allnames, names

def load_from_json(file_path: Path):
    assert file_path.suffix == ".json"
    with open(file_path, encoding="UTF-8") as file:
        return json.load(file)

def computeNearFar(mesh,masks, intrinsics, image_dims, poses, choose_poses = None, epsilon = 2e-1, onlyNear = False, path = None, vaxnerfstyle = False):
    if vaxnerfstyle:
        onlyNear = False
        epsilon = 0.0
    nearsDict = {}
    farsDict = {}
    if choose_poses is None:
        choose_poses = range(len(poses))
    # mesh.vertices = mesh.vertices * self.bound
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)
    print("Getting Nears and Fars")
    for poseidx in tqdm(choose_poses):
        intrinsic = torch.from_numpy(intrinsics[poseidx]).float()
        img_dims = image_dims[poseidx]
        pose = torch.from_numpy(poses[poseidx]).float()
        i,j = torch.meshgrid(torch.arange(img_dims[0]), torch.arange(img_dims[1]))
        zs = torch.ones_like(i)
        points = torch.stack((i, j, zs), dim=-1).to(dtype=torch.float32)
        directions = points @ intrinsic.inverse().transpose(1, 0)

        rays_d = directions @ pose[ :3, :3].transpose(-1, -2) # (B, N, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = pose[:3, 3].unsqueeze(0) # [B, 3]
        rays_o = rays_o[:, :].expand_as(rays_d) # [B, N, 3]
        rays_o = rays_o.view(-1,rays_o.shape[-1]).numpy()
        rays_d = rays_d.view(-1,rays_d.shape[-1]).numpy()
        multiple_hits= not onlyNear
        if not onlyNear:
            iLocs, iIdxs , _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)
            
            iDists = torch.norm(torch.from_numpy(iLocs) - rays_o[iIdxs], dim=-1).float()

            nears = -1*torch.ones_like(torch.tensor(rays_o[:,-1])).unsqueeze(-1)

            nears[iIdxs,0] = iDists -epsilon

            iLocs, iIdxs , _ = intersector.intersects_location(rays_o + 10*rays_d, -rays_d, multiple_hits=False)
            
            iDists = torch.norm(torch.from_numpy(iLocs) - rays_o[iIdxs], dim=-1).float()

            
            fars = -1*torch.ones_like(torch.tensor(rays_o[:,-1])).unsqueeze(-1)
            fars[iIdxs,0] = iDists + epsilon
        else:
            iLocs, iIdxs , _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)
            # visualize_poses(poses, mesh = mesh)
            
            iDists = torch.norm(torch.from_numpy(iLocs) - rays_o[iIdxs], dim=-1).float()
            nears = -1*torch.ones_like(torch.tensor(rays_o[:,-1])).unsqueeze(-1)
            nears[iIdxs,0] = iDists  - 0.5*epsilon
            fars = nears.clone()
            fars[iIdxs,0]  = iDists + epsilon

            # nears[iIdxs,0] = iDists  - epsilon
            # fars = nears.clone()
            # fars[iIdxs,0]  = iDists + 2*epsilon


            # iLocs, iIdxs , _ = intersector.intersects_location(rays_o + 10*rays_d, -rays_d, multiple_hits=False)
            
            # iDists = torch.norm(torch.from_numpy(iLocs) - rays_o[iIdxs], dim=-1).float()

            
            # fars = -1*torch.ones_like(torch.tensor(rays_o[:,-1])).unsqueeze(-1)
            # fars[iIdxs,0] = iDists
            # import pdb; pdb.set_trace()

        # nears = -1*torch.ones_like(torch.tensor(rays_o[:,-1])).unsqueeze(-1)
        # nears[iIdxs] = torch.tensor(iDists).float().unsqueeze(-1) - epsilon
        # fars = nears.clone() + 2*epsilon
        nears_curr = nears.reshape(int(img_dims[0]),int(img_dims[1])).transpose(0,1)
        fars_curr = fars.reshape(int(img_dims[0]),int(img_dims[1])).transpose(0,1)
        # import pdb; pdb.set_trace()
        # nears_curr, fars_curr = self.filterPixels(nears_curr,fars_curr,masks[poseidx])
        # img = np.ones((int(intrinsics[5]),int(intrinsics[4])))*0
        # img = img.reshape(-1,1)
        # img[iIdxs] = 255.0
        # img = img.reshape(int(intrinsics[4]),int(intrinsics[5])).transpose(1,0)
        # cv2.imshow(str(poseidx),img)
        # cv2.imwrite("testmask.png",img.astype('uint8'))
        # cv2.imwrite("testmaskgt.png",masks[poseidx].astype('uint8'))


        # imgvis = np.ones((int(intrinsics[5]),int(intrinsics[4])))*0
        # imgvis[fars_curr == -1] = 255.0
        # imgvis[np.where(masks[poseidx]==0)] = 0
        # cv2.imwrite("shittypixels1.png",imgvis.astype('uint8'))
        nearsDict[poseidx] = nears_curr.numpy()
        farsDict[poseidx] = fars_curr.numpy()
    if path is not None:
        np.savez(path,nears=nearsDict,fars=farsDict)
    return nearsDict, farsDict
def visualize_poses(poses, size=0.1, bound=2, mesh=None, segcolors=None):
    # poses: [B, 4, 4]
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(bound, bound, bound)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(poses.shape[0]):
        pose = poses[i]
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 0.5

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        # import pdb; pdb.set_trace()
        if segcolors is not None:
            for e in segs.entities:
                e.color = [segcolors[i][0],segcolors[i][1],segcolors[i][2],255]
        objects.append(segs)
    if mesh is not None:
        mesh = trimesh.load_mesh(mesh)
        mesh.visual.vertex_colors = trimesh.visual.random_color()
        objects.append(mesh)

    trimesh.Scene(objects).show(viewer='gl')
    return trimesh.Scene(objects)
def computeVisualHull(masks, poses, bound, intrinsics, image_dims, outpath, min_views_visual_hull=2,config=None):
    
    res = 512 #256
    x = torch.linspace(-bound, bound, res)
    y = torch.linspace(-bound, bound, res)
    z = torch.linspace(-bound, bound, res)
    xv, yv, zv = torch.meshgrid(x, y, z)
    pts = torch.stack((xv, yv, zv, torch.ones_like(xv)), dim=-1).view((-1, 4)).T
    # occupancy = torch.ones_like(xv).view(-1)
    # visibility = torch.ones_like(xv).view(-1)
    occupancy = torch.zeros_like(xv).view(-1)
    visibility = torch.zeros_like(xv).view(-1)
    count = 0
    poses = torch.from_numpy(poses).float()
    # visualize_poses(poses)
    pts_mask = torch.norm(pts[:3, :], dim=0) <= 1.0
    intrinsics = torch.from_numpy(intrinsics).float()
    for intrinsic, image_dim, pose, mask in zip(intrinsics, image_dims, poses, masks):
        # import pdb; pdb.set_trace()
        count += 1
        K = intrinsic
        pts_cam = torch.matmul(pose.inverse(), pts)[:-1, :]
        pts_mask_negDepth = pts_cam[2, :]>=0
        pts_img = torch.matmul(K, pts_cam) / pts_cam[2, :]
        pts_img = torch.round(pts_img).long()
        pts_idx = (pts_cam[2, :] > 0) & (pts_img[0,:]>=0) & (pts_img[0,:] < image_dim[0]) & (pts_img[1,:]>=0) & (pts_img[1,:] < image_dim[1]) & pts_mask_negDepth
        # import pdb; pdb.set_trace()
        if config.vaxnerf:
            print("Using Vaxnerf style mask")
            maskscurrent =  cv2.dilate(mask.astype(np.uint8), np.ones((7,7)), iterations=1)
        else:
            if config.mask_dilate > 0:
                print(f"Dilating mask with kernel size {config.mask_dilate} for visual hull")
                kernel = np.ones((config.mask_dilate, config.mask_dilate), dtype=np.uint8)
                maskscurrent = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            else:
                maskscurrent = mask
        occupancy[pts_idx] += torch.from_numpy(maskscurrent[pts_img[1, pts_idx], pts_img[0, pts_idx]]>0)
        visibility[pts_idx] += 1
    # min(self.poses.shape[0],2)
    # self.min_views_visual_hull = 2
    
    # vhull: bool = False  # If True, use visual hull for nears and fars
    # vhull_vanilla: bool = False  # If True, use visual hull for nears and fars
    # vhull_unitsph: bool = False  # If True, use visual hull for nears and fars
    # vhull_scc: bool = False  # If True, use visual hull for nears and fars
    if config is not None:
        ourvhull = config.vhull
        ourvhull = ourvhull & (not config.vhull_vanilla) & (not config.vhull_unitsph)
        unitspherecull = config.vhull_unitsph
        scc = config.vhull_scc
    else:
        ourvhull = True
        unitspherecull = False
        scc = config.vhull_scc
    if not ourvhull:
        occupancy = (occupancy > 0) & (occupancy >= visibility)
        if unitspherecull:
            occupancy = occupancy & pts_mask
    else:
        if config.vhull_onlyvd:
            occupancy  = (visibility>min(poses.shape[0]-1, min_views_visual_hull)) 
        else:
            occupancy  = (visibility>min(poses.shape[0]-1, min_views_visual_hull)) & (occupancy == visibility)
    

    verts, faces, normals, values = skimage.measure.marching_cubes(occupancy.float().numpy().reshape(res, res, res), level=0.5, spacing=[bound*2.0/(res-1)] * 3)

    verts = verts-bound
    mesh = trimesh.Trimesh(vertices=verts,faces=faces,)
    if scc and not config.vhull_onlyvd:
        outfile_unclean = str(outpath)+"_unclean.ply"
    else:
        print("Not cleaning vhull mesh")
        outfile_unclean = str(outpath)+"_clean.ply"
    mesh.export(outfile_unclean, file_type='ply')
    if scc and not config.vhull_onlyvd:
        print("Cleaning vhull mesh")
        ccIdx = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        maxId = np.argmax(np.bincount(ccIdx))
        mesh.update_faces(ccIdx==maxId)
        mesh.export(str(outpath)+"_clean.ply", file_type='ply')
    # import pdb; pdb.set_trace()
    return mesh

def orbit_poses(nposes, radii=[1], theta_range=[0, 2*np.pi/3], phi_range=[0, 2*np.pi], random=False):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    if len(radii) == 1:
        orbit_radii = [radii[0]]
    else:
        assert radii[0] < radii[1], "Passed radii range [min, max]"
        orbit_radii = np.linspace(radii[0], radii[1], 1)

    size = nposes // len(orbit_radii)
    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(nposes, 1, 1)
    for i, radius in enumerate(orbit_radii):
        if random:
            if len(theta_range) > 1:
                thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
            else:
                thetas = torch.tensor(theta_range).repeat(size).to(dtype=torch.float32)

            if len(phi_range) > 1:
                phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
            else:
                phis = torch.tensor(phi_range).repeat(size).to(dtype=torch.float32)
        else:
            if len(theta_range) > 1:
                thetas = torch.linspace(theta_range[0], theta_range[1], size)
            else:
                thetas = torch.tensor(theta_range).repeat(size).to(dtype=torch.float32)

            if len(phi_range) > 1:
                phis = torch.linspace(phi_range[0], phi_range[1], size)
            else:
                phis = torch.tensor(phi_range).repeat(size).to(dtype=torch.float32)

        print(phis)


        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

        # lookat
        forward_vector = -normalize(centers)
        up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
        right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

        poses[i*size:(i+1)*size, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
        poses[i*size:(i+1)*size, :3, 3] = centers

    poses = poses.cpu().numpy()
    return poses
class ActorsHQ(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            print("Rendering test poses")
        meta_path = Path(config.meta_exp)
        base_path = Path(config.data_dir)
        actor = meta_path.parents[0].name
        data_path = base_path / actor / 'Sequence1' / '4x' 
        conf_path = meta_path / 'conf.json'

        vhull_path = Path(config.exp_path) / 'vhull'
        if self.split.value == "train":
            nearsandfars_path = Path(config.exp_path) / 'nearsAndFars.npz'
        else:
            nearsandfars_path = Path(config.exp_path) / 'nearsAndFars_val.npz'
        m = load_from_json(conf_path)
        if self.split.value == "train":
            cams = m['train_cameras']
            timestamp_path = meta_path / 'splits' / 'train.txt'
        else:
            cams = m['val_cameras']
            timestamp_path = meta_path / 'splits' / 'test.txt'
        cams = [int(x) for x in cams.split(' ')]
        self.orgcamlen = len(cams)
        f = open(timestamp_path,'r')
        timestamp = f.read()
        f.close()
        poses_all, intrinsics_all, image_dims, allposes, allintrinsics, allimgdims, allnames, names = load_actorshq(data_path, cams, timestamp, scalefactor = 1.0)
        cam2pix = intrinsics_all
        poses = poses_all
        

        # visualize_poses(np.vstack([newposes,poses]), segcolors = np.vstack([np.array([[255,0,0]]*len(newposes)),np.array([[0,255,0]]*len(poses))]), mesh = None)
        # import pdb; pdb.set_trace()




        image_width = image_dims[:,0]
        image_height = image_dims[:,1]
        self.imageNames = names
        images = []        
        masks = []
        normals = []
        imgpath = data_path / 'rgbs'
        maskpath = data_path / 'masks'
        trainpath = str(Path(config.exp_path) / 'trainimages')
        os.makedirs(trainpath, exist_ok=True)
        for i in tqdm(range(len(names))):
            frame = names[i]
            fname =str((imgpath / frame)  / (frame+f'_rgb{int(timestamp):06d}.jpg'))
            mname =str((maskpath / frame)  / (frame+f'_mask{int(timestamp):06d}.png'))
            rgb = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB).astype('float')
            try:
                mask = cv2.imread(mname, cv2.IMREAD_UNCHANGED).astype('float')
            except:
                mask = np.ones_like(rgb[...,0])*255
            mask = mask>200.0
            rgb[mask == 0,...] = 255.0
            if self.split.value == "train":
                cv2.imwrite(str(Path(trainpath) / frame) + '_rgb.png',rgb.astype('uint8'))
                cv2.imwrite(str(Path(trainpath) / frame) + '_mask.png',mask.astype('uint8')*255)

            # import pdb; pdb.set_trace()
            get_norml = (compnormals.getNormals(rgb.astype('uint8')).astype('float')/255.0)*2.0-1.0
            normals.append(get_norml)
            rgb = rgb/255.0
            images.append(rgb)
            masks.append(mask)
        if (self.split.value == "train" or (self.split.value == "test" and config.vaxnerf)) and (config.vhull or config.vhull_vanilla or config.vhull_unitsph):
            if not os.path.isfile(nearsandfars_path):
                if not os.path.isfile(str(vhull_path)+'_clean.ply'):
                    vhull_mesh = computeVisualHull(masks, poses,2.0, cam2pix, image_dims, vhull_path, min_views_visual_hull=config.vhullK,config=config)
                else:
                    print("Loading visual hull")
                    vhull_mesh = trimesh.load(str(vhull_path)+'_clean.ply',process=False)
                nears, fars = computeNearFar(vhull_mesh,masks, cam2pix, image_dims, poses, onlyNear = True,path=nearsandfars_path,vaxnerfstyle=config.vaxnerf)
            else:
                dataloadnf = np.load(nearsandfars_path,allow_pickle=True)
                nears = dataloadnf['nears'][None][0]
                fars = dataloadnf['fars'][None][0]
            for key in nears: nears[key][nears[key]==-1] = self.near
            for key in fars: fars[key][fars[key]==-1] = self.far
        else:
            nears = [img[...,0]*0+self.near for img in images]
            fars = [img[...,0]*0+self.far for img in images]


        if self.split.value == "train":
            from scipy.spatial.transform import Rotation
            from scipy.spatial.transform import Slerp
            intset = set()
            for i in range(len(poses)):
                pose_dists = np.linalg.norm(poses[:,:3,3]-poses[i,:3,3],axis=-1)
                pose_idxs = np.argsort(pose_dists)
                pose_idxs = pose_idxs[1:]
                intset.add(tuple(sorted((i,pose_idxs[0]))))
                intset.add(tuple(sorted((i,pose_idxs[1]))))
            
            newposes = []
            numposesperpair = 1
            for i,j in intset:
                # import pdb; pdb.set_trace()
                key_rots = Rotation.from_matrix(poses[[i,j],:3,:3])
                key_times = [0.0,1.0]
                slerp = Slerp(key_times, key_rots)
                slerptimes = np.linspace(0,1,numposesperpair+1,endpoint=False)[1:]
                interp_rots = slerp(slerptimes)
                interp_trans = [ ((1-t)*poses[i,:3,3]+ t*poses[j,:3,3]) for t in slerptimes]
                interp_poses = np.zeros((len(interp_rots),4,4))
                interp_poses[:,:3,:3] = interp_rots.as_matrix()
                interp_poses[:,:3,3] = interp_trans
                interp_poses[:,3,3] = 1.0
                newposes.append(interp_poses)
            newposes = np.vstack(newposes)
            # os.makedirs(str(Path(config.exp_path) / 'newmask'), exist_ok=True)
            vhull_mesh = trimesh.load(str(vhull_path)+'_clean.ply',process=False)
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(vhull_mesh, scale_to_box=False)
            newmasks = []
            newimgs = []
            newintrinsics = []
            newheights = []
            newwidths = []
            newnormals = []
            intrinsic = torch.from_numpy(allintrinsics[0]).float()
            img_dims = torch.from_numpy(allimgdims[0]).float()
            camnum = 0
            for camnum in range(newposes.shape[0]):
                pose = torch.from_numpy(newposes[camnum]).float()
                i,j = torch.meshgrid(torch.arange(img_dims[0]), torch.arange(img_dims[1]))
                zs = torch.ones_like(i)
                points = torch.stack((i, j, zs), dim=-1).to(dtype=torch.float32)
                directions = points @ intrinsic.inverse().transpose(1, 0)

                rays_d = directions @ pose[ :3, :3].transpose(-1, -2) # (B, N, 3)
                rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                rays_o = pose[:3, 3].unsqueeze(0) # [B, 3]
                rays_o = rays_o[:, :].expand_as(rays_d) # [B, N, 3]
                rays_o = rays_o.view(-1,rays_o.shape[-1]).numpy()
                rays_d = rays_d.view(-1,rays_d.shape[-1]).numpy()

                _, iIdxs , _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)
                mask = torch.zeros_like(torch.tensor(rays_o[:,-1])).unsqueeze(-1)
                mask[iIdxs,0] = 1.0
                mask = mask.reshape(int(img_dims[0]),int(img_dims[1])).transpose(0,1)
                newmasks.append(mask.numpy())
                newimgs.append(images[0]*0-1)
                newintrinsics.append(allintrinsics[0])
                newheights.append(image_dims[0][1])
                newwidths.append(image_dims[0][0])
                newnormals.append(normals[0]*0-1)
                # cv2.imwrite(f'{camnum}_mask.png',mask.numpy().astype('uint8')*255)
                # camnum += 1
                # cv2.imwrite(str(Path(config.exp_path) / 'newmask' / cimname) + '_mask.png',mask.numpy().astype('uint8')*255)
            poses = np.vstack([poses,newposes])
            masks = masks + newmasks
            images = images + newimgs
            image_height = np.hstack([image_height,newheights])
            image_width = np.hstack([image_width,newwidths])
            normals = normals + newnormals
            cam2pix = np.vstack([cam2pix,newintrinsics])
            cidx = len(nears)
            for ii in range(len(newposes)):
                nears[cidx+ii] = nears[0]*0 + self.near
                fars[cidx+ii] = fars[0]*0 + self.far


        #visualize_poses(poses_all, mesh=str(vhull_path)+'_clean.ply')
        self.nears = nears
        self.fars = fars
        self.images = images
        self.normals = normals
        self.masks = masks
        self.height, self.width = image_height,image_width
        self.camtoworlds = poses
        self.focal = cam2pix[:,0,0]
        self.pixtocams = np.linalg.inv(cam2pix)


        if config.render_path:
            orbit_cams = orbit_poses(40, radii=[2.0 * (1.0 if config.vhull else 2.3)], theta_range=[2*np.pi / 3- np.pi/6, 2*np.pi / 3- np.pi/6], phi_range=[2*np.pi, 0], random=False)
            # from scipy.spatial.transform import Rotation as R
            # r = R.from_rotvec([0, 0, 0]).as_matrix()
            # R2 = np.eye(4)
            # R2[:3,:3] = r
            # orbit_cams = orbit_cams@R2
            orbit_cams[:,:3,3] = orbit_cams[:,:3,3] - np.mean(orbit_cams[:,:3,3],axis=0) + np.mean(allposes[:,:3,3],axis=0)
            camidx = 2
            cam2pix = cam2pix[camidx][None,...].repeat(orbit_cams.shape[0],0)
            self.nears = [nears[camidx] for id in range(orbit_cams.shape[0])]
            self.fars =  [fars[camidx] for id in range(orbit_cams.shape[0])]
            self.images = [images[camidx] for id in range(orbit_cams.shape[0])]
            self.normals = [normals[camidx] for id in range(orbit_cams.shape[0])]
            self.masks = [masks[camidx] for id in range(orbit_cams.shape[0])]
            self.height, self.width = [image_height[camidx] for id in range(orbit_cams.shape[0])],[image_width[camidx] for id in range(orbit_cams.shape[0])]
            self.camtoworlds = orbit_cams
            self.focal = cam2pix[:,0,0]
            self.pixtocams = np.linalg.inv(cam2pix)
            # visualize_poses(orbit_cams)

            
    def _make_ray_batch(self,
                        pix_x_int,
                        pix_y_int,
                        cam_idx,
                        lossmult=None
                        ):
        """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """
        # print("NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        # print(cam_idx)
        # print(pix_x_int)
        # print(pix_y_int)
        # print(len(self.images))
        # print(len(self.camtoworlds))
        broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
        ray_kwargs = {
            'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
            'cam_idx': broadcast_scalar(cam_idx),
        }
        # print(cam_idx)
        # Collect per-camera information needed for each ray.
        if self.metadata is not None:
            # Exposure index and relative shutter speed, needed for RawNeRF.
            for key in ['exposure_idx', 'exposure_values']:
                idx = 0 if self.render_path else cam_idx
                ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
        if self.exposures is not None:
            idx = 0 if self.render_path else cam_idx
            ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
        if self.render_path and self.render_exposures is not None:
            ray_kwargs['exposure_values'] = broadcast_scalar(
                self.render_exposures[cam_idx])

        pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

        # Slow path, do ray computation using numpy (on CPU).
        batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)
        batch['cam_dirs'] = -self.camtoworlds[ray_kwargs['cam_idx'][..., 0]][..., :3, 2]

        # import trimesh
        # pts = batch['origins'][..., None, :] + batch['directions'][..., None, :] * np.linspace(0, 1, 5)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test.ply", "ply")
        #
        # pts = batch['origins'][0, 0, None, :] - self.camtoworlds[cam_idx][:, 2] * np.linspace(0, 1, 100)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test2.ply", "ply")

        # print("687: " + str(cam_idx))
        # sys.stdout.flush() 
        # if not self.split == utils.DataSplit.TRAIN:
        #     import pdb; pdb.set_trace()
        if 1: #not self.render_path:
            if (not isinstance(cam_idx, list)) and (not isinstance(cam_idx, np.ndarray)):
                batch['rgb']= self.images[cam_idx][pix_y_int,pix_x_int]
                batch['normals'] = self.normals[cam_idx][pix_y_int,pix_x_int]
                batch['mask'] = self.masks[cam_idx][pix_y_int,pix_x_int]
                batch['near']= self.nears[cam_idx][pix_y_int,pix_x_int]
                batch['far']= self.fars[cam_idx][pix_y_int,pix_x_int]

                if self.config.vaxnerf:
                    batch['nears_vh'] = batch['near']
                    batch['fars_vh'] = batch['far']
                    batch['near']= batch['nears_vh']*0 + self.near
                    batch['far']= batch['fars_vh']*0 + self.far

            else:
                cam_idx_sq = cam_idx.squeeze() #squeeze(-1).squeeze(-1)
                pix_y_int_sq = pix_y_int #.squeeze()
                pix_x_int_sq = pix_x_int #.squeeze()
                # import pdb; pdb.set_trace()
                batch['rgb'] = np.stack([self.images[cam_idx_sq[idx]][pix_y_int_sq[idx], pix_x_int_sq[idx]] for idx in range(0,cam_idx_sq.shape[0])],axis=0)
                # import pdb; pdb.set_trace()
                batch['rgb']= batch['rgb'].reshape([*pix_x_int_sq.shape,3])
                batch['mask'] = np.stack([self.masks[cam_idx_sq[idx]][pix_y_int_sq[idx], pix_x_int_sq[idx]] for idx in range(0,cam_idx_sq.shape[0])],axis=0)
                batch['mask']= batch['mask'].reshape([*pix_x_int_sq.shape,1])

                batch['normals'] = np.stack([self.normals[cam_idx_sq[idx]][pix_y_int_sq[idx], pix_x_int_sq[idx]] for idx in range(0,cam_idx_sq.shape[0])],axis=0)
                batch['normals']= batch['normals'].reshape([*pix_x_int_sq.shape,3])

                batch['near'] = np.stack([self.nears[cam_idx_sq[idx]][pix_y_int_sq[idx], pix_x_int_sq[idx]] for idx in range(0,cam_idx_sq.shape[0])],axis=0)
                batch['near']= batch['near'].reshape([*pix_x_int_sq.shape,1])
                batch['far'] = np.stack([self.fars[cam_idx_sq[idx]][pix_y_int_sq[idx], pix_x_int_sq[idx]] for idx in range(0,cam_idx_sq.shape[0])],axis=0)
                batch['far']= batch['far'].reshape([*pix_x_int_sq.shape,1])

                if self.config.vaxnerf:
                    batch['nears_vh'] = batch['near']
                    batch['fars_vh'] = batch['far']
                    batch['near']= batch['nears_vh']*0 + self.near
                    batch['far']= batch['fars_vh']*0 + self.far

        if self._load_disps:
            batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
        # if self._load_normals:
        #     batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
        #     batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_train(self, item):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        # self._patch_size = 2
        num_patches = self._batch_size // self._patch_size ** 2
        # import pdb; pdb.set_trace()
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
        else:
            cam_idx = np.random.randint(0, self._n_examples, (1,))
            
        lower_border = self._num_border_pixels_to_mask
        upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
        # Random pixel patch x-coordinates.
        cam_idx_sq = cam_idx.squeeze(-1).squeeze(-1)
        # if num_patches == 1:
        #     cam_idx_sq = [cam_idx_sq]
        pix_x_int = np.array([np.random.randint(lower_border, self.width[id] - upper_border,(1)).item() for id in cam_idx_sq])
        # Random pixel patch y-coordinates.
        pix_y_int = np.array([np.random.randint(lower_border, self.height[id] - upper_border,(1)).item() for id in cam_idx_sq])
        pix_x_int = pix_x_int.reshape(cam_idx.shape)
        pix_y_int = pix_y_int.reshape(cam_idx.shape)
        # Add patch coordinate offsets.
        # Shape will broadcast to (num_patches, _patch_size, _patch_size).
        patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
            self._patch_size, self._patch_size)
        pix_x_int = pix_x_int + patch_dx_int
        pix_y_int = pix_y_int + patch_dy_int

        if self._apply_bayer_mask:
            # Compute the Bayer mosaic mask for each pixel in the batch.
            lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
        else:
            lossmult = None

        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                    lossmult=lossmult)

    def generate_ray_batch(self, cam_idx: int):
        """Generate ray batch for a specified camera in the dataset."""
        if 0:#self._render_spherical:
            camtoworld = self.camtoworlds[cam_idx]
            rays = camera_utils.cast_spherical_rays(
                camtoworld, self.height[cam_idx], self.width[cam_idx], self.near, self.far)
            return rays
        else:
            # Generate rays for all pixels in the image.
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                self.width[cam_idx], self.height[cam_idx])
            return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

class LLFF(Dataset):
    """LLFF Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        # Set up scaling factor.
        image_dir_suffix = ''
        # Use downsampling factor (unless loading training split for raw dataset,
        # we train raw at full resolution because of the Bayer mosaic pattern).
        if config.factor > 0 and not (config.rawnerf_mode and
                                      self.split == utils.DataSplit.TRAIN):
            image_dir_suffix = f'_{config.factor}'
            factor = config.factor
        else:
            factor = 1

        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = os.path.join(self.data_dir, 'sparse/0/')

        # Load poses.
        if utils.file_exists(colmap_dir):
            pose_data = NeRFSceneManager(colmap_dir).process()
        else:
            # # Attempt to load Blender/NGP format if COLMAP data not present.
            # pose_data = load_blender_posedata(self.data_dir)
            raise ValueError('COLMAP data not found.')
        image_names, poses, pixtocam, distortion_params, camtype = pose_data

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        poses = poses[inds]

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocam = pixtocam @ np.diag([factor, factor, 1.])
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal = 1. / self.pixtocams[0, 0]
        self.distortion_params = distortion_params
        self.camtype = camtype

        # Separate out 360 versus forward facing scenes.
        if config.forward_facing:
            # Set the projective matrix defining the NDC transformation.
            self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = camera_utils.recenter_poses(poses)
            self.colmap_to_world_transform = (
                    transform @ self.colmap_to_world_transform)
            # Forward-facing spiral render path.
            self.render_poses = camera_utils.generate_spiral_path(
                poses, bounds, n_frames=config.render_path_frames)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform = camera_utils.transform_poses_pca(poses)
            self.colmap_to_world_transform = transform
            if config.render_spline_keyframes is not None:
                rets = camera_utils.create_render_spline_path(config, image_names,
                                                              poses, self.exposures)
                self.spline_indices, self.render_poses, self.render_exposures = rets
            else:
                # Automatically generated inward-facing elliptical render path.
                self.render_poses = camera_utils.generate_ellipse_path(
                    poses,
                    n_frames=config.render_path_frames,
                    z_variation=config.z_variation,
                    z_phase=config.z_phase)

        # Select the split.
        all_indices = np.arange(len(image_names))
        if config.llff_use_all_images_for_training:
            train_indices = all_indices
        else:
            train_indices = all_indices % config.llffhold != 0
        if config.llff_use_all_images_for_testing:
            test_indices = all_indices
        else:
            test_indices = all_indices % config.llffhold == 0
        split_indices = {
            utils.DataSplit.TEST: all_indices[test_indices],
            utils.DataSplit.TRAIN: all_indices[train_indices],
        }
        indices = split_indices[self.split]
        image_names = [image_names[i] for i in indices]
        poses = poses[indices]
        # if self.split == utils.DataSplit.TRAIN:
        #     # load different training data on different rank
        #     local_indices = [i for i in range(len(image_names)) if (i + self.global_rank) % self.world_size == 0]
        #     image_names = [image_names[i] for i in local_indices]
        #     poses = poses[local_indices]
        #     indices = local_indices

        raw_testscene = False
        if config.rawnerf_mode:
            # Load raw images and metadata.
            images, metadata, raw_testscene = raw_utils.load_raw_dataset(
                self.split,
                self.data_dir,
                image_names,
                config.exposure_percentile,
                factor)
            self.metadata = metadata

        else:
            # Load images.
            colmap_image_dir = os.path.join(self.data_dir, 'images')
            image_dir = os.path.join(self.data_dir, 'images' + image_dir_suffix)
            for d in [image_dir, colmap_image_dir]:
                if not utils.file_exists(d):
                    raise ValueError(f'Image folder {d} does not exist.')
            # Downsampled images may have different names vs images used for COLMAP,
            # so we need to map between the two sorted lists of files.
            colmap_files = sorted(utils.listdir(colmap_image_dir))
            image_files = sorted(utils.listdir(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, colmap_to_image[f])
                           for f in image_names]
            images = [utils.load_img(x) for x in tqdm(image_paths, desc='Loading LLFF dataset', disable=self.global_rank != 0, leave=False)]
            images = np.stack(images, axis=0) / 255.

            # EXIF data is usually only present in the original JPEG images.
            jpeg_paths = [os.path.join(colmap_image_dir, f) for f in image_names]
            exifs = [utils.load_exif(x) for x in jpeg_paths]
            self.exifs = exifs
            if 'ExposureTime' in exifs[0] and 'ISOSpeedRatings' in exifs[0]:
                gather_exif_value = lambda k: np.array([float(x[k]) for x in exifs])
                shutters = gather_exif_value('ExposureTime')
                isos = gather_exif_value('ISOSpeedRatings')
                self.exposures = shutters * isos / 1000.

        if raw_testscene:
            # For raw testscene, the first image sent to COLMAP has the same pose as
            # the ground truth test image. The remaining images form the training set.
            raw_testscene_poses = {
                utils.DataSplit.TEST: poses[:1],
                utils.DataSplit.TRAIN: poses[1:],
            }
            poses = raw_testscene_poses[self.split]

        self.poses = poses
        self.images = images
        self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = images.shape[1:3]


class TanksAndTemplesNerfPP(Dataset):
    """Subset of Tanks and Temples Dataset as processed by NeRF++."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            split_str = 'camera_path'
        else:
            split_str = self.split.value

        basedir = os.path.join(self.data_dir, split_str)

        # TODO: need to rewrite this to put different data on different rank
        def load_files(dirname, load_fn, shape=None):
            files = [
                os.path.join(basedir, dirname, f)
                for f in sorted(utils.listdir(os.path.join(basedir, dirname)))
            ]
            mats = np.array([load_fn(utils.open_file(f, 'rb')) for f in files])
            if shape is not None:
                mats = mats.reshape(mats.shape[:1] + shape)
            return mats

        poses = load_files('pose', np.loadtxt, (4, 4))
        # Flip Y and Z axes to get correct coordinate frame.
        poses = np.matmul(poses, np.diag(np.array([1, -1, -1, 1])))

        # For now, ignore all but the first focal length in intrinsics
        intrinsics = load_files('intrinsics', np.loadtxt, (4, 4))

        if not config.render_path:
            images = load_files('rgb', lambda f: np.array(Image.open(f))) / 255.
            self.images = images
            self.height, self.width = self.images.shape[1:3]

        else:
            # Hack to grab the image resolution from a test image
            d = os.path.join(self.data_dir, 'test', 'rgb')
            f = os.path.join(d, sorted(utils.listdir(d))[0])
            shape = utils.load_img(f).shape
            self.height, self.width = shape[:2]
            self.images = None

        self.camtoworlds = poses
        self.focal = intrinsics[0, 0, 0]
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)


class TanksAndTemplesFVS(Dataset):
    """Subset of Tanks and Temples Dataset as processed by Free View Synthesis."""

    def _load_renderings(self, config):
        """Load images from disk."""
        render_only = config.render_path and self.split == utils.DataSplit.TEST

        basedir = os.path.join(self.data_dir, 'dense')
        sizes = [f for f in sorted(utils.listdir(basedir)) if f.startswith('ibr3d')]
        sizes = sizes[::-1]

        if config.factor >= len(sizes):
            raise ValueError(f'Factor {config.factor} larger than {len(sizes)}')

        basedir = os.path.join(basedir, sizes[config.factor])
        open_fn = lambda f: utils.open_file(os.path.join(basedir, f), 'rb')

        files = [f for f in sorted(utils.listdir(basedir)) if f.startswith('im_')]
        if render_only:
            files = files[:1]
        images = np.array([np.array(Image.open(open_fn(f))) for f in files]) / 255.

        names = ['Ks', 'Rs', 'ts']
        intrinsics, rot, trans = (np.load(open_fn(f'{n}.npy')) for n in names)

        # Convert poses from colmap world-to-cam into our cam-to-world.
        w2c = np.concatenate([rot, trans[..., None]], axis=-1)
        c2w_colmap = np.linalg.inv(camera_utils.pad_poses(w2c))[:, :3, :4]
        c2w = c2w_colmap @ np.diag(np.array([1, -1, -1, 1]))

        # Reorient poses so z-axis is up
        poses, _ = camera_utils.transform_poses_pca(c2w)
        self.poses = poses

        self.images = images
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = poses
        # For now, ignore all but the first focal length in intrinsics
        self.focal = intrinsics[0, 0, 0]
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)

        if render_only:
            render_path = camera_utils.generate_ellipse_path(
                poses,
                config.render_path_frames,
                z_variation=config.z_variation,
                z_phase=config.z_phase)
            self.images = None
            self.camtoworlds = render_path
            self.render_poses = render_path
        else:
            # Select the split.
            all_indices = np.arange(images.shape[0])
            indices = {
                utils.DataSplit.TEST:
                    all_indices[all_indices % config.llffhold == 0],
                utils.DataSplit.TRAIN:
                    all_indices[all_indices % config.llffhold != 0],
            }[self.split]

            self.images = self.images[indices]
            self.camtoworlds = self.camtoworlds[indices]


class DTU(Dataset):
    """DTU Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the DTU dataset.')

        images = []
        pixtocams = []
        camtoworlds = []

        # Find out whether the particular scan has 49 or 65 images.
        n_images = len(utils.listdir(self.data_dir)) // 8

        # Loop over all images.
        for i in range(1, n_images + 1):
            # Set light condition string accordingly.
            if config.dtu_light_cond < 7:
                light_str = f'{config.dtu_light_cond}_r' + ('5000'
                                                            if i < 50 else '7000')
            else:
                light_str = 'max'

            # Load image.
            fname = os.path.join(self.data_dir, f'rect_{i:03d}_{light_str}.png')
            image = utils.load_img(fname) / 255.
            if config.factor > 1:
                image = lib_image.downsample(image, config.factor)
            images.append(image)

            # Load projection matrix from file.
            fname = os.path.join(self.data_dir, f'../../cal18/pos_{i:03d}.txt')
            with utils.open_file(fname, 'rb') as f:
                projection = np.loadtxt(f, dtype=np.float32)

            # Decompose projection matrix into pose and camera matrix.
            camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
            camera_mat = camera_mat / camera_mat[2, 2]
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_mat.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]
            pose = pose[:3]
            camtoworlds.append(pose)

            if config.factor > 0:
                # Scale camera matrix according to downsampling factor.
                camera_mat = np.diag([1. / config.factor, 1. / config.factor, 1.
                                      ]).astype(np.float32) @ camera_mat
            pixtocams.append(np.linalg.inv(camera_mat))

        pixtocams = np.stack(pixtocams)
        camtoworlds = np.stack(camtoworlds)
        images = np.stack(images)

        def rescale_poses(poses):
            """Rescales camera poses according to maximum x/y/z value."""
            s = np.max(np.abs(poses[:, :3, -1]))
            out = np.copy(poses)
            out[:, :3, -1] /= s
            return out

        # Center and scale poses.
        camtoworlds, _ = camera_utils.recenter_poses(camtoworlds)
        camtoworlds = rescale_poses(camtoworlds)
        # Flip y and z axes to get poses in OpenGL coordinate system.
        camtoworlds = camtoworlds @ np.diag([1., -1., -1., 1.]).astype(np.float32)

        all_indices = np.arange(images.shape[0])
        split_indices = {
            utils.DataSplit.TEST: all_indices[all_indices % config.dtuhold == 0],
            utils.DataSplit.TRAIN: all_indices[all_indices % config.dtuhold != 0],
        }
        indices = split_indices[self.split]

        self.images = images[indices]
        self.height, self.width = images.shape[1:3]
        self.camtoworlds = camtoworlds[indices]
        self.pixtocams = pixtocams[indices]


class Multicam(Dataset):
    def __init__(self,
                 split: str,
                 data_dir: str,
                 config: configs.Config):
        super().__init__(split, data_dir, config)

        self.multiscale_levels = config.multiscale_levels

        images, camtoworlds, pixtocams, pixtocam_ndc = \
            self.images, self.camtoworlds, self.pixtocams, self.pixtocam_ndc
        self.heights, self.widths, self.focals, self.images, self.camtoworlds, self.pixtocams, self.lossmults = [], [], [], [], [], [], []
        if pixtocam_ndc is not None:
            self.pixtocam_ndc = []
        else:
            self.pixtocam_ndc = None

        for i in range(self._n_examples):
            for j in range(self.multiscale_levels):
                self.heights.append(self.height // 2 ** j)
                self.widths.append(self.width // 2 ** j)

                self.pixtocams.append(pixtocams @ np.diag([self.height / self.heights[-1],
                                                           self.width / self.widths[-1],
                                                           1.]))
                self.focals.append(1. / self.pixtocams[-1][0, 0])
                if config.forward_facing:
                    # Set the projective matrix defining the NDC transformation.
                    self.pixtocam_ndc.append(pixtocams.reshape(3, 3))

                self.camtoworlds.append(camtoworlds[i])
                self.lossmults.append(2. ** j)
                self.images.append(self.down2(images[i], (self.heights[-1], self.widths[-1])))
        self.pixtocams = np.stack(self.pixtocams)
        self.camtoworlds = np.stack(self.camtoworlds)
        self.cameras = (self.pixtocams,
                        self.camtoworlds,
                        self.distortion_params,
                        np.stack(self.pixtocam_ndc) if self.pixtocam_ndc is not None else None)
        self._generate_rays()

        if self.split == utils.DataSplit.TRAIN:
            # Always flatten out the height x width dimensions
            def flatten(x):
                if x[0] is not None:
                    x = [y.reshape([-1, y.shape[-1]]) for y in x]
                    if self._batching == utils.BatchingMethod.ALL_IMAGES:
                        # If global batching, also concatenate all data into one list
                        x = np.concatenate(x, axis=0)
                    return x
                else:
                    return None

            self.batches = {k: flatten(v) for k, v in self.batches.items()}
        self._n_examples = len(self.camtoworlds)

        # Seed the queue with one batch to avoid race condition.
        if self.split == utils.DataSplit.TRAIN:
            self._next_fn = self._next_train
        else:
            self._next_fn = self._next_test

    def _generate_rays(self):
        if self.global_rank == 0:
            tbar = tqdm(range(len(self.camtoworlds)), desc='Generating rays', leave=False)
        else:
            tbar = range(len(self.camtoworlds))

        self.batches = defaultdict(list)
        for cam_idx in tbar:
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                self.widths[cam_idx], self.heights[cam_idx])
            broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
            ray_kwargs = {
                'lossmult': broadcast_scalar(self.lossmults[cam_idx]),
                'near': broadcast_scalar(self.near),
                'far': broadcast_scalar(self.far),
                'cam_idx': broadcast_scalar(cam_idx),
            }

            pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

            batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)
            if not self.render_path:
                batch['rgb'] = self.images[cam_idx]
            if self._load_disps:
                batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
            if self._load_normals:
                batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
                batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
            for k, v in batch.items():
                self.batches[k].append(v)

    def _next_train(self, item):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = self._batch_size // self._patch_size ** 2
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            ray_indices = np.random.randint(0, self.batches['origins'].shape[0], (num_patches, 1, 1))
            batch = {k: v[ray_indices] if v is not None else None for k, v in self.batches.items()}
        else:
            image_index = np.random.randint(0, self._n_examples, ())
            ray_indices = np.random.randint(0, self.batches['origins'][image_index].shape[0], (num_patches,))
            batch = {k: v[image_index][ray_indices] if v is not None else None for k, v in self.batches.items()}
        batch['cam_dirs'] = -self.camtoworlds[batch['cam_idx'][..., 0]][..., 2]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_test(self, item):
        """Sample next test batch (one full image)."""
        batch = {k: v[item] for k, v in self.batches.items()}
        batch['cam_dirs'] = -self.camtoworlds[batch['cam_idx'][..., 0]][..., 2]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    @staticmethod
    def down2(img, sh):
        return cv2.resize(img, sh[::-1], interpolation=cv2.INTER_CUBIC)


class MultiLLFF(Multicam, LLFF):
    pass


if __name__ == '__main__':
    from internal import configs
    import accelerate

    config = configs.Config()
    accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    config.factor = 8
    dataset = LLFF('test', '/SSD_DISK/datasets/360_v2/bicycle', config)
    print(len(dataset))
    for _ in tqdm(dataset):
        pass
    print('done')
    # print(accelerator.process_index)
