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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, CameraInfo, generate_ellipse_path_from_camera_infos
import time
import torch
import numpy as np
import skimage, trimesh

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], extra_opts=None, load_ply=None, interpCams=False, load_best=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # type: ignore
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration and load_ply is None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), best=load_best)
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.render_cameras = {}
        self.vid_cameras = {}

        if hasattr(extra_opts, 'use_dust3r') and extra_opts.use_dust3r: # type: ignore
            scene_info = sceneLoadTypeCallbacks["DUSt3R"](args.source_path, args.images, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "sparse")): # type: ignore
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "transforms_alignz_train.json")): # type: ignore
            print("Found transforms_alignz_train.json file, assuming OpenIllumination data set!")
            scene_info = sceneLoadTypeCallbacks["OpenIllumination"](args.source_path, args.white_background, args.eval, extra_opts=extra_opts) # type: ignore
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and load_ply is None:
            # NOTE :this dump use the file name, we dump the SceneInfo.pcd as the input.ply
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.render_cameras:
                camlist.extend(scene_info.render_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            init_time = time.time()
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, mode="train")
            init_time2 = time.time()
            print("Loading training cameras with {}s".format(init_time2 - init_time))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            init_time3 = time.time()
            print("Loading test cameras with {}s".format(time.time() - init_time2))
            self.render_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.render_cameras, resolution_scale, args)
            print("Loading render cameras with {}s".format(time.time() - init_time3))

        if self.loaded_iter:
            load_name = "point_cloud.ply"
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), load_name))
        elif load_ply:
            self.gaussians.load_ply(load_ply)
            # in this case, we need it to be trainable, so we need to make sure the spatial_lr_scale is not 0
            self.gaussians.spatial_lr_scale = self.cameras_extent
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.save_ply(os.path.join(self.model_path, "input.ply"))

        #add training cameras to this list
        if interpCams:
            self.addInterpPoses(args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getAllCameras(self, scale=1.0):
        return self.train_cameras[scale] + self.test_cameras[scale]

    def getRenderCameras(self, scale=1.0):
        return self.render_cameras[scale]

    def getInterpCameras(self, scale=1.0):
        return self.inter_cameras[scale]

    def getVidCameras(self, scale=1.0):
        return self.vid_cameras[scale]
        
    def addInterpPoses(self,args):
        import math
        def fov2focal(fov, pixels):
            return pixels / (2 * math.tan(fov / 2))
        def focal2fov(focal, pixels):
            return pixels / (2 * math.tan(fov / 2))

        cam_locations = []
        cam_rotations = []
        cam_T = []
        Ts = []
        Ks = []
        images = []
        masks = []
        
        for cam_info in self.train_cameras[1.0]:
            cam_locations.append(cam_info.camera_center)
            cam_rotations.append(cam_info.R)
            cam_T.append(cam_info.T)
            Ts.append(cam_info.world_view_transform.T)
            fx = cam_info.FX
            fy = cam_info.FY
            cx = cam_info.CX
            cy = cam_info.CY
            Ks.append(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
            images.append(cam_info.original_image)
            masks.append(cam_info.mask)
        Ts = torch.stack(Ts,dim=0)
        Ks = torch.stack(Ks,dim=0)
        mesh, poses, meanT, scaleT = computeVisualHull(masks, Ts, 2.0, Ks, min_views_visual_hull=3)
        
        from scipy.spatial.transform import Rotation
        from scipy.spatial.transform import Slerp
        intset = set()
        nearestGTcams = []
        for i in range(len(poses)):
            pose_dists = np.linalg.norm(poses[:,:3,3]-poses[i,:3,3],axis=-1)
            pose_idxs = np.argsort(pose_dists)
            pose_idxs = pose_idxs[1:]
            nearestGTcams.append(pose_idxs[:2])
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
            nearestcamsCurr = []
            for interpTrans in interp_trans:
                pose_dists = np.linalg.norm(poses[:,:3,3]-interpTrans,axis=-1)
                pose_idxs = np.argsort(pose_dists)
                pose_idxs = pose_idxs[1:3]
                nearestcamsCurr.append(pose_idxs)
            interp_poses = np.zeros((len(interp_rots),4,4))
            interp_poses[:,:3,:3] = interp_rots.as_matrix()
            interp_poses[:,:3,3] = interp_trans
            interp_poses[:,3,3] = 1.0
            newposes.append(interp_poses)
            for nearcam in nearestcamsCurr:
                nearestGTcams.append(nearcam.tolist())
        newposes = np.vstack(newposes)
        nearestGTcams = np.vstack(nearestGTcams)
        # os.makedirs(str(Path(config.exp_path) / 'newmask'), exist_ok=True)
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

        avgarea = 0
        for mask in masks:
            avgarea += mask.sum()
        avgarea /= len(masks)

        newmasks = []
        newimgs = []
        newintrinsics = []
        intrinsic = Ks[0].float()
        img_dims = torch.tensor([masks[0].shape[2],masks[0].shape[1]])
        camnum = 0
        goodposes = []
        finnearestGTcams = []
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
            # if not mask.sum() > 0.5*avgarea:
            #     continue
            mask = mask.reshape(int(img_dims[0]),int(img_dims[1])).transpose(0,1)

            goodposes.append(newposes[camnum])
            finnearestGTcams.append(nearestGTcams[camnum])
            newmasks.append(mask.numpy())
            newimgs.append(mask[0]*0-1)
            newintrinsics.append(Ks[0])
        
        # visualize_poses(torch.vstack([poses,torch.from_numpy(np.stack(goodposes,axis=0))]).cpu().numpy())
        goodposes = torch.from_numpy(np.stack(goodposes,axis=0)).float().to(Ts.device)
        scaleT = scaleT.float().to(Ts.device)
        meanT = meanT.float().to(Ts.device)
        newmasks = [torch.from_numpy(mask).float().to(Ts.device) for mask in newmasks]
        newcameras = []
        # import pdb; pdb.set_trace()
        for i in range(len(goodposes)):
            cam_info = CameraInfo(
                uid=f'{i+len(Ts):d}',
                R=goodposes[i,:3,:3].cpu().numpy(),
                T=(-(goodposes[i,:3,:3].T@(goodposes[i,:3,3]*scaleT + meanT))).cpu().numpy(),
                FovY=self.train_cameras[1.0][0].FoVy,
                FovX=self.train_cameras[1.0][0].FoVx,
                CX=self.train_cameras[1.0][0].CX,
                CY=self.train_cameras[1.0][0].CY,
                FX=self.train_cameras[1.0][0].FX,
                FY=self.train_cameras[1.0][0].FY,
                image=self.train_cameras[1.0][0].original_image*0+1,
                image_path='none.jpg',
                image_name='frame_pose_'+f'{i:d}',	
                width=newmasks[i].shape[1],
                height=newmasks[i].shape[0],
                mask=newmasks[i].reshape(*masks[0].shape),
                mono_depth=None,
                confidence=None,
                is_dust3r=False
            )
            new_cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                        FoVx=cam_info.FovX, FoVy=cam_info.FovY, CX=cam_info.CX, CY=cam_info.CY, FX=cam_info.FX, FY=cam_info.FY,
                        image=cam_info.image, gt_alpha_mask=cam_info.mask, mono_depth=cam_info.mono_depth,
                        image_name=cam_info.image_name, uid=len(Ts)+i, 
                        data_device=self.train_cameras[1.0][0].data_device, white_background=args.white_background)
            newcameras.append(new_cam)
            # import pdb; pdb.set_trace()
        self.inter_cameras = {}
        self.inter_cameras[1.0] = newcameras

    def addVidPoses(self, args, n_frames=120, const_speed=True, z_variation=0., z_phase=0.):
        """Create a circular video path around the object from the training cameras."""
        train_cam_infos = []
        for cam in self.train_cameras[1.0]:
            train_cam_infos.append(CameraInfo(
                uid=cam.uid,
                R=cam.R,
                T=cam.T,
                FovY=cam.FoVy,
                FovX=cam.FoVx,
                CX=cam.CX,
                CY=cam.CY,
                FX=cam.FX,
                FY=cam.FY,
                image=np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8),
                image_path='',
                image_name=cam.image_name,
                width=cam.image_width,
                height=cam.image_height,
                mask=None,
                mono_depth=None,
                confidence=None,
                is_dust3r=False,
            ))

        vid_cam_infos = generate_ellipse_path_from_camera_infos(
            train_cam_infos,
            n_frames=n_frames,
            const_speed=const_speed,
            z_variation=z_variation,
            z_phase=z_phase,
        )

        self.vid_cameras = {}
        self.vid_cameras[1.0] = []
        for vid_cam_info in vid_cam_infos:
            blank_img = torch.zeros((3, vid_cam_info.height, vid_cam_info.width), dtype=torch.float32)
            self.vid_cameras[1.0].append(Camera(
                colmap_id=vid_cam_info.uid,
                R=vid_cam_info.R,
                T=vid_cam_info.T,
                FoVx=vid_cam_info.FovX,
                FoVy=vid_cam_info.FovY,
                FX=vid_cam_info.FX,
                FY=vid_cam_info.FY,
                CX=vid_cam_info.CX,
                CY=vid_cam_info.CY,
                image=blank_img,
                gt_alpha_mask=None,
                mono_depth=None,
                image_name=vid_cam_info.image_name,
                uid=vid_cam_info.uid,
                data_device=self.train_cameras[1.0][0].data_device,
                white_background=args.white_background,
            ))



def computeVisualHull(masks, poses, bound, intrinsics, min_views_visual_hull=2):
    
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
    poses = poses.cpu().float()
    posesnew = torch.zeros_like(poses)
    for i in range(len(poses)):
        posesnew[i,:3,3] = -(poses[i,:3,:3].T)@poses[i,:3,3]
        posesnew[i,:3,:3] = poses[i,:3,:3].T
        posesnew[i,3,3] = 1.0 
    poses = posesnew
    meanT = poses[:,:3,3].mean(0)
    poses[:,:3,3] -= meanT
    scaleT = (poses[:,:3,3].norm(dim=-1)).max()
    poses[:,:3,3] /= scaleT
    
    # visualize_poses(poses.numpy())
    intrinsics = intrinsics.cpu().float()
    for intrinsic, pose, mask in zip(intrinsics, poses, masks):
        # import pdb; pdb.set_trace()
        mask = mask.squeeze().cpu()
        image_dim = [mask.shape[1],mask.shape[0]]
        count += 1
        K = intrinsic
        # pose[:3,:3] = pose[:3,:3].T
        pts_cam = torch.matmul(pose.inverse(), pts)[:-1, :]
        pts_mask_negDepth = pts_cam[2, :]>=0
        pts_img = torch.matmul(K, pts_cam) / pts_cam[2, :]
        pts_img = torch.round(pts_img).long()
        pts_idx = (pts_cam[2, :] > 0) & (pts_img[0,:]>=0) & (pts_img[0,:] < image_dim[0]) & (pts_img[1,:]>=0) & (pts_img[1,:] < image_dim[1]) & pts_mask_negDepth
        maskscurrent = mask
        occupancy[pts_idx] += maskscurrent[pts_img[1, pts_idx], pts_img[0, pts_idx]]>0
        visibility[pts_idx] += 1
        
    occupancy  = (visibility>min(poses.shape[0]-1, min_views_visual_hull)) & (occupancy == visibility)
    
    verts, faces, normals, values = skimage.measure.marching_cubes(occupancy.float().numpy().reshape(res, res, res), level=0.5, spacing=[bound*2.0/(res-1)] * 3)

    verts = verts-bound
    mesh = trimesh.Trimesh(vertices=verts,faces=faces,)
    print("Cleaning vhull mesh")
    ccIdx = trimesh.graph.connected_component_labels(mesh.face_adjacency)
    maxId = np.argmax(np.bincount(ccIdx))
    mesh.update_faces(ccIdx==maxId)
    return mesh, poses, meanT, scaleT

def visualize_poses(poses, size=0.1, bound=2, mesh = None, segcolors=None):
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