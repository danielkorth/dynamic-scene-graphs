import os
import cv2
import numpy as np
from typing import List

from utils.open3d_utils import CameraPose

def load_all_color_images(color_dir: str) -> List[np.ndarray]:
    """Load all color images from the color directory as numpy arrays."""
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(color_dir, fname), cv2.IMREAD_COLOR) for fname in color_files]
    return images

def load_all_depth_images(depth_dir: str) -> List[np.ndarray]:
    """Load all depth images from the depth directory as numpy arrays."""
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    images = [cv2.imread(os.path.join(depth_dir, fname), cv2.IMREAD_UNCHANGED) for fname in depth_files]
    return images

def read_trajectory(filename) -> List[CameraPose]:
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

def unproject_depth_to_world_grid(depth: np.ndarray, K: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Unproject a depth image to a structured (H, W, 3) grid of 3D world coordinates.
    Invalid (zero) depth pixels will be set to np.nan.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pixels_hom = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)
    K_inv = np.linalg.inv(K)
    rays = pixels_hom @ K_inv.T  # (H, W, 3)
    points_cam = rays * depth[..., None]  # (H, W, 3)
    # Transform to world coordinates
    points_world = (rotation @ points_cam.reshape(-1, 3).T).T + translation  # (H*W, 3)
    points_world = points_world.reshape(H, W, 3)
    # Set invalid points to nan
    points_world[depth == 0] = np.nan
    return points_world