import os
import json
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from einops import rearrange

# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
def read_camera(folder):
    """
    read camera from json file
    """
    scene_info = json.load(open(os.path.join(folder, 'camera_data.json')))
    try:
        intrinsics = scene_info['intrinsics']
    except:
        pass

    rgb_files = []
    poses = []
    max_depths = []
    for item in scene_info['frames']:
        rgb_files.append(os.path.join(folder, item['file_path'].strip("depth") + "rgb.png"))

        # World to agent transform
        w2a = np.array(item['transform_matrix'])

        # Agent to world transform
        a2h = np.eye(4)
        a2h[:3, 3] = np.array([0, 1.5, 0])

        # Habitat coords to camera coords
        h2c = np.eye(4)
        h2c[1,1] = -1
        h2c[2,2] = -1 

        # Camera to world
        c2w = w2a @ a2h @ h2c

        poses.append(c2w)
        max_depths.append(np.array(item['max_depth']))
    return rgb_files, poses, intrinsics, max_depths

def read_all(folder, include_semantics=False):
    """
    read source images from a folder
    """
    src_rgb_files, src_poses, intrinsics, max_depths = read_camera(folder)

    src_cameras = []
    src_rgbs = []
    src_depths = []
    src_semantics = []

    for src_rgb_file, src_pose, max_depth in zip(src_rgb_files, src_poses, max_depths):
        src_rgb , src_depth, src_semantic, src_camera = \
        read_image(src_rgb_file, 
                   src_pose, 
                   max_depth=max_depth,
                   intrinsics=intrinsics,
                   include_semantics=include_semantics)

        src_rgbs.append(src_rgb)
        src_depths.append(src_depth)
        src_cameras.append(src_camera)
        src_semantics.append(src_semantic)
    
    src_depths = torch.stack(src_depths, axis=0)
    src_rgbs = torch.stack(src_rgbs, axis=0)
    src_semantics = torch.stack(src_semantics, axis=0)
    src_cameras = torch.stack(src_cameras, axis=0)

    return {
        "rgb": src_rgbs[..., :3],
        "camera": src_cameras,
        "depth": src_depths,
        "semantic": src_semantics
    }


def read_image(rgb_file, pose, max_depth, intrinsics, include_semantics=False):
    rgb = torch.from_numpy(imageio.imread(rgb_file).astype(np.float32) / 255.0)
    depth = torch.from_numpy(imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32) / 255.0 * max_depth)
    semantic = torch.from_numpy(imageio.imread(rgb_file[:-7]+'semantic.png').astype(np.float32) / 255.0)

    image_size = rgb.shape[:2]
    intrinsic = np.eye(4)
    intrinsic[:3,:3] = np.array(intrinsics)

    camera = torch.from_numpy(np.concatenate(
        (list(image_size), intrinsic.flatten(), pose.flatten())
    ).astype(np.float32))

    return rgb, depth, semantic, camera