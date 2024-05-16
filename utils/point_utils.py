import os
import json
import torch
import random
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from utils.data_utils import read_all
from typing import BinaryIO, Dict, List, Optional, Union


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return H, W, intrinsics, c2w

def get_rays_single_image(H, W, intrinsics, c2w, render_stride=1):
        """
        :param H: image height H
        :param W: image width W
        :param intrinsics: 4 by 4 intrinsic matrix [B, 4, 4]
        :param c2w: 4 by 4 camera to world extrinsic matrix [B, 4, 4]
        :return: rays_o, rays_d [B, HxW, 3] [B, HxW, 3]
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        intrinsics = intrinsics.to(device)
        c2w = c2w.to(device)

        # indexing = x, y
        u, v = np.meshgrid(np.arange(W)[:: render_stride], np.arange(H)[:: render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels).to(device)
        batched_pixels = pixels.unsqueeze(0).repeat(len(intrinsics), 1, 1)
        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)

        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[1], 1)
        )  # B x HW x 3

        return rays_o, rays_d


def get_point_cloud(path, include_semantics=False):
    """
    Depth images to point cloud
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = read_all(path, include_semantics=include_semantics)
    cameras = data["camera"]

    rgbs = data["rgb"].to(device)
    depths = data["depth"].to(device)
    semantics = data["semantic"].to(device)

    Hs, Ws, intrinsics, c2ws = parse_camera(cameras)
    W, H = int(Ws[0].item()), int(Hs[0].item())

    rays_o, rays_d = get_rays_single_image(H=H, W=W, intrinsics=intrinsics, c2w=c2ws)

    pts = rays_o + rays_d * depths.flatten(1).unsqueeze(-1)
    coords = pts.cpu().numpy()
    rgbs = rgbs.flatten(1,-2).cpu().numpy()
    semantics = 255 * semantics.flatten(1).squeeze().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(rgbs.reshape(-1, 3))
    pcd.estimate_normals()
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.075)

    point_path = os.path.join(path, "points3d.ply")
    o3d.io.write_point_cloud(point_path, downsampled_pcd)

    return
