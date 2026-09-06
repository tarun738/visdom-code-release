import argparse
import json
import math
import os
import time
from argparse import Namespace

import camtools as ct
import numpy as np
import open3d as o3d
import torch
from tqdm import trange
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from torch.nn import functional as F
import copy
from typing import NamedTuple
from torchvision import transforms

class SceneInfo(NamedTuple):
    Ks: list
    Ts: list
    images: list
    masks: list
    widths: list
    heights: list


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def points2homopoints(points):
    assert points.shape[-1] == 3
    bottom = torch.ones_like(points[...,0:1])
    return torch.cat([points, bottom], dim=-1)

def batch_projection(Ks, Ts, points):
    '''
    Ks: B, 3, 3
    Ts: B, 4, 4
    points: B, N, 3
    '''
    pre_fix = points.shape[:-1] # [100, 100]
    points = points.reshape(-1, 3) # [M, 3]

    Ts = torch.stack(Ts, dim=0) # [N, 4, 4]
    Ks = torch.stack(Ks, dim=0).to(Ts.device) # [N, 3, 3]
    camera_num = Ks.shape[0]
    homopts = points2homopoints(points) # [M, 4]
    # world to camera # [N, M, 4] @ [N, 4, 4] = [N, M, 4]
    homopts_cam = torch.bmm(homopts.unsqueeze(0).repeat_interleave(Ts.shape[0], dim=0), Ts.transpose(1,2)) 
    # camera to image space  # [N, M, 4] @ [N, 4, 3] = [N, M, 3]
    homopts_img = torch.bmm(homopts_cam[...,:3], Ks.transpose(1,2))
    # normalize
    homopts_img = homopts_img / (homopts_img[...,2:] + 1e-6)
    # reshape back
    homopts_img = homopts_img.reshape(camera_num, *pre_fix, 3)
    homopts_cam = homopts_cam.reshape(camera_num, *pre_fix, 4)
    return homopts_img[...,0:2], homopts_cam[...,2]

def query_from_list_with_list(listA: list, listB: list):
    '''
    listA: [1, 2, 3]
    listB: [3, 2, 1]
    return: [2, 1, 0]
    '''
    return [listB[i] for i in listA]

def simple_resize_image(img, size):
    return transforms.Resize(size, antialias=True)(img)

# Below this many points the saved hull is not usable as a gaussian initialisation.
# Only checked on the final pass: the first pass carves a coarse grid over a large box
# purely to locate the object, and legitimately returns very few points for a small
# object (guitar_002 yields 14, then 22848 in the final pass).
MIN_HULL_POINTS = 100


def get_visual_hull(N, bbox, scene_info, cam_center, usevisibledomain=False,
                    vd_K=None, occ_ratio=1.0, min_points=1):
    """Carve a visual hull on an N^3 grid spanning bbox.

    vd_K is K from the paper: keep a voxel only if at least this many views observe it.
    None means len(views) - 1, the default used for the paper's results.
    occ_ratio is the fraction of observing views that must also see the voxel inside the
    silhouette. Counts are integers, so values above 0.95 only separate from 1.0 at 20 or
    more views.
    """
    t_start = time.time()
    pcs = []
    pcsvd = []
    color = []
    all_pts = []
    Ks = scene_info.Ks
    Ts = scene_info.Ts
    images = scene_info.images
    masks = scene_info.masks
    # Per-camera original image dimensions. These must not be taken from a single
    # camera: the frustum test and the uv normalisation below are in each camera's
    # own pixel coordinates, and a scene may mix image sizes.
    widths = torch.tensor(scene_info.widths, dtype=torch.float32).view(-1, 1, 1).cuda()
    heights = torch.tensor(scene_info.heights, dtype=torch.float32).view(-1, 1, 1).cuda()

    [xs, ys, zs], [xe, ye, ze] = bbox[0], bbox[1]

    # please note that in vasedeck, the images are not same size, for simplify, just resize them
    new_images = []
    new_masks = []
    img_size = images[0].shape[1:]
    for image, mask in zip(images, masks):
        new_images.append(simple_resize_image(image, img_size))
        new_masks.append(simple_resize_image(mask, img_size))

    images = torch.stack(new_images) # N C H W
    masks = torch.stack(new_masks) # N 1 H W

    for h_id in trange(N):
        i, j = torch.meshgrid(torch.linspace(xs, xe, N).cuda(),
                              torch.linspace(ys, ye, N).cuda(), indexing='ij')
        i, j = i.t(), j.t()
        pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1)
        pts[...,2] = h_id / N * (ze - zs) + zs # 100, 100, 3

        # shift the pts to be centered at the camera center
        pts[...,0] += cam_center[0]  # note the order, [x, y, z], width, height, depth
        pts[...,1] += cam_center[1]
        pts[...,2] += cam_center[2]

        all_pts.append(pts)

        # now we have the pts, we need to project them to the image plane
        # batched projection
        uv, z = batch_projection(Ks, Ts, pts) # [N, 100, 100, 2], [N, 100, 100]
        valid_z_mask = z > 0
        valid_x_y_mask = (uv[...,0] > 0) & (uv[...,0] < widths) & (uv[...,1] > 0) & (uv[...,1] < heights)
        valid_pt_mask = valid_z_mask & valid_x_y_mask

        # simple resize the uv to [-1, 1]
        uv[...,0] = uv[...,0] / widths * 2 - 1
        uv[...,1] = uv[...,1] / heights * 2 - 1

        # now we have the uv, we use grid_sample to sample the image to get the color
        result = F.grid_sample(images.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 3
        # Sample the mask nearest-neighbour: a silhouette is a binary decision per pixel,
        # and bilinear interpolation followed by a > 0 test would count any partial overlap
        # at the boundary as inside. This matches the zipnerf pipeline, which indexes the
        # mask with rounded pixel coordinates.
        result_mask = F.grid_sample(masks.float(), uv, mode='nearest', padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 1

        valid_pt_mask_final = (result_mask.squeeze() > 0) & valid_pt_mask
        if not usevisibledomain:
            pcs.append((valid_pt_mask_final.float().sum(0) >= (images.shape[0] - 1 if vd_K is None else min(vd_K, images.shape[0])))) # [100, 100]
        else:
            pcs.append(valid_pt_mask_final.float().sum(0)) # [100, 100]
            pcsvd.append(valid_pt_mask.float().sum(0))
        color.append(result.mean(0)) # [100, 100, 3]
    
    pcs = torch.stack(pcs, -1)
    if usevisibledomain:
        pcsvd = torch.stack(pcsvd,-1)
    color = torch.stack(color, -1)

    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    n_views = images.shape[0]
    K = n_views - 1 if vd_K is None else min(vd_K, n_views)
    if usevisibledomain:
        # pcs counts views seeing the voxel inside the silhouette, pcsvd counts views
        # observing it at all. pcs <= pcsvd holds by construction, so occ_ratio 1.0 is
        # exactly the previous equality test.
        idx = torch.where((pcs >= occ_ratio * pcsvd) & (pcsvd >= K))
    else:
        idx = torch.where(pcs > 0)

    color = torch.stack((r[idx] * 255, g[idx] * 255, b[idx] * 255), -1)

    idx = torch.stack([idx[1], idx[0], idx[2]], -1) # note the order is hwz -> xyz
    # turn the idx to the point position used in batch_projection
    idx = idx.float() / N
    idx[...,0] = idx[...,0] * (xe - xs) + xs + cam_center[0]
    idx[...,1] = idx[...,1] * (ye - ys) + ys + cam_center[1]
    idx[...,2] = idx[...,2] * (ze - zs) + zs + cam_center[2]

    if idx.shape[0] < min_points:
        raise ValueError(
            f"Visual hull carved only {idx.shape[0]} points on an {N}^3 grid, below the "
            f"minimum of {min_points}. The grid spans {bbox[0]} to {bbox[1]} around "
            "the mean camera position; nothing rescales the scene into it. Check "
            "--cube_size and the --cube_size_shift_* offsets, and that the masks are "
            "non-empty. A very small --vd_K can also do this, by keeping distant voxels "
            "in the first pass and so inflating the box the second pass is computed over.")
    print("visual hull is Okay, with {} points".format(idx.shape[0]))
    print(f"[vhull] grid {N}^3, {n_views} views, K={K if usevisibledomain else 'n/a'}, "
          f"occ_ratio={occ_ratio}: {time.time() - t_start:.2f}s")
    # we get the point cloud, use open3d to visualize it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(idx.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy() / 255)

    # get bbox
    bbox = pcd.get_axis_aligned_bounding_box()
    return pcd, bbox


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='generate k views covering object')
    parser.add_argument('--data_dir', type=str, default='sparse_nerf_datasets/sparse_omni3d_undistorted/backpack_016', help='data directory, we only support colmap type data, kitchen, garden')
    parser.add_argument("--cube_size", type=float, default=4.0,
                        help="half-extent of the first-pass carving box, in scene units")
    parser.add_argument("--voxel_num", type=int, default=200,
                        help="side length of the first-pass carving grid (voxels per axis)")
    parser.add_argument("--voxel_num_final", type=int, default=64,
                        help="side length of the second-pass carving grid, which produces the "
                             "saved point cloud (voxels per axis)")
    parser.add_argument('--sparse_id', type=int, default=-1, help='sparse id')
    parser.add_argument('--reso', type=int, default=1,
                        help='image downscale factor passed to the camera loader; the released '
                             'run scripts pass 2 for omni3d and mip360, 1 for ActorsHQ')
    parser.add_argument('--vd', action='store_true', help='use the visible domain constraint')
    parser.add_argument('--vd_K', type=int, default=None,
                        help='K from the paper: keep a voxel only if at least this many views '
                             'observe it. Default (unset) is (number of views - 1), the '
                             'value used for the paper\'s results.')
    parser.add_argument('--save_bbox', type=str, default=None,
                        help='write the second-pass bounding box to this JSON file')
    parser.add_argument('--load_bbox', type=str, default=None,
                        help='read the second-pass bounding box from this JSON file instead of '
                             'deriving it from the first pass. Required for a meaningful K ablation: '
                             'a smaller K retains distant voxels, which enlarges the derived box '
                             'and coarsens the fixed-resolution second pass, so hulls built at '
                             'different K are otherwise not comparable.')
    parser.add_argument('--occ_ratio', type=float, default=0.95,
                        help='fraction of observing views that must also see the voxel inside '
                             'the silhouette. Vote counts are integers, so this only differs '
                             'from 1.0 once a voxel is seen by 20 or more views.')
    parser.add_argument('--not_vis', action='store_true', help='whether vis the visual hull, is enable, not vis')
    parser.add_argument("--cube_size_shift_x", type=float, default=0.0, help="shift sizex of the cube in meters")
    parser.add_argument("--cube_size_shift_y", type=float, default=0.0, help="shift sizey of the cube in meters")
    parser.add_argument("--cube_size_shift_z", type=float, default=0.0, help="shift sizez of the cube in meters")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extra_opts = Namespace()
    extra_opts.sparse_view_num = -1
    extra_opts.resolution = args.reso
    extra_opts.use_mask = True
    extra_opts.data_device = 'cuda'
    extra_opts.init_pcd_name = 'origin'
    extra_opts.white_background = False

    # load the camera parameters
    # we assume that the camera parameters are stored in the data_dir
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.data_dir, 'images', False, extra_opts=extra_opts) 
    camlist = cameraList_from_camInfos(scene_info.train_cameras, 1.0, extra_opts)

    # if sparse id is not zero, we only use given frames to construct the visual hull
    if args.sparse_id >= 0:
        selected_id = np.loadtxt(os.path.join(args.data_dir, f"sparse_{str(args.sparse_id)}.txt"), dtype=np.int32)
        print("the sparse id is {}, with {} frames".format(args.sparse_id, len(selected_id)))
        assert args.sparse_id == len(selected_id)
    else:
        selected_id = np.arange(len(camlist))

    # get all camera locations to recenter the scene
    cam_locations = []
    cam_rotations = []
    cam_T = []
    Ts = []
    Ks = []
    images = []
    masks = []
    widths = []
    heights = []
    for cam_info in camlist:

        cam_locations.append(cam_info.camera_center)
        cam_rotations.append(cam_info.R)
        cam_T.append(cam_info.T)
        Ts.append(cam_info.world_view_transform.T)
        fx = cam_info.FX
        fy = cam_info.FY
        cx = cam_info.CX
        cy = cam_info.CY
        Ks.append(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float())
        images.append(cam_info.original_image)
        masks.append(cam_info.mask)
        widths.append(cam_info.image_width)
        heights.append(cam_info.image_height)

    # in this time, we already have the camera parameters
    # first, we get the cemera locations center
    cam_center = torch.stack(cam_locations).mean(0)
    print('the camera center is:', cam_center)
 
    Ks = query_from_list_with_list(selected_id, Ks)
    Ts = query_from_list_with_list(selected_id, Ts)
    images = query_from_list_with_list(selected_id, images)
    masks = query_from_list_with_list(selected_id, masks)
    widths = query_from_list_with_list(selected_id, widths)
    heights = query_from_list_with_list(selected_id, heights)

    scene_info = SceneInfo(Ks, Ts, images, masks, widths, heights)
    Ks_clone = copy.deepcopy(Ks)

    bx = args.cube_size
    init_bbox = [[args.cube_size_shift_x-bx, args.cube_size_shift_y-bx, args.cube_size_shift_z-bx], 
                 [args.cube_size_shift_x+bx, args.cube_size_shift_y+bx, args.cube_size_shift_z+bx]]
    if args.load_bbox:
        with open(args.load_bbox) as f:
            saved = json.load(f)
        enlarged_bbox_min = np.array(saved["min"])
        enlarged_bbox_max = np.array(saved["max"])
        print(f"Loaded second-pass bbox from {args.load_bbox}")
    else:
        # we run the get_visual_hull twice, first to get the bound, second to get the visual hull
        pcd, bbox = get_visual_hull(args.voxel_num, init_bbox, scene_info, cam_center,
                                    usevisibledomain=args.vd, vd_K=args.vd_K, occ_ratio=args.occ_ratio)
    
    # since we get the bound, we use this bound to better recon
    # we use the center of the bound as the center of the scene
    # please note that the bbox may need bigger, since the camera may not cover the whole scene
        bbox_min = bbox.get_min_bound()
        bbox_max = bbox.get_max_bound()
        # Calculate the center point of the original bounding box
        center = (bbox_min + bbox_max) / 2
        # Calculate the extents of the original bounding box
        extents = bbox_max - bbox_min
        # Double the box around its centre
        scale_factor = 2
        # Calculate the scaled extents
        scaled_extents = extents * scale_factor
        # Calculate the new minimum and maximum points of the enlarged bounding box
        enlarged_bbox_min = center - scaled_extents / 2
        enlarged_bbox_max = center + scaled_extents / 2

    if args.save_bbox:
        with open(args.save_bbox, "w") as f:
            json.dump({"min": list(map(float, enlarged_bbox_min)),
                       "max": list(map(float, enlarged_bbox_max))}, f)
        print(f"Wrote second-pass bbox to {args.save_bbox}")

    pcd, bbox_new = get_visual_hull(args.voxel_num_final, [enlarged_bbox_min, enlarged_bbox_max],
                                    scene_info, [0,0,0], usevisibledomain=args.vd,
                                    vd_K=args.vd_K, occ_ratio=args.occ_ratio,
                                    min_points=MIN_HULL_POINTS)
    # save the pointcloud
    if args.sparse_id >= 0:
        # Only non-default K / occ_ratio get a suffix, so the default filenames -- which
        # train_gs.py and render.py refer to via --init_pcd_name -- stay unchanged.
        suffix = ""
        if args.vd_K is not None:
            suffix += f"_K{args.vd_K}"
        if args.occ_ratio != parser.get_default('occ_ratio'):
            suffix += f"_occ{args.occ_ratio:g}"
        if args.vd:
            o3d.io.write_point_cloud(os.path.join(args.data_dir, f"visual_hull_{str(args.sparse_id)}_vd{suffix}.ply"), pcd)
        else:
            o3d.io.write_point_cloud(os.path.join(args.data_dir, f"visual_hull_{str(args.sparse_id)}{suffix}.ply"), pcd)
    else:
        o3d.io.write_point_cloud(os.path.join(args.data_dir, "visual_hull_full.ply"), pcd)

    if not args.not_vis:
        Ts = np.array([i.cpu().numpy() for i in Ts])
        Ks = np.array(Ks_clone)
        cameras = ct.camera.create_camera_frames(Ks, Ts, highlight_color_map={0: [1, 0, 0], -1: [0, 1, 0]})
        # build LineSet to represent the coordinate
        world_coord = o3d.geometry.LineSet()
        world_coord.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [2, 0, 0], 
                                                                [0, 0, 0], [0, 2, 0], 
                                                                [0, 0, 0], [0, 0, 2]]))
        world_coord.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 3], [0, 5]]))
        # X->red, Y->green, Z->blue
        world_coord.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        # Optional: some COLMAP datasets have no triangulated points. actorshq_to_colmap.py
        # writes an empty points3D.txt by design, since gaussians start from the hull.
        pcd_path = os.path.join(args.data_dir, "sparse/0/points3D.ply")
        pcdo = o3d.io.read_point_cloud(pcd_path) if os.path.isfile(pcd_path) else None

        # init viewer
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(cameras)
        viewer.add_geometry(pcd)
        viewer.add_geometry(world_coord)
    
        opt = viewer.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()
