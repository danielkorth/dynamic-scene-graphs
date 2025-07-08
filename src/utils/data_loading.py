import os
import cv2
import numpy as np
from typing import List

from utils.open3d_utils import CameraPose

COLOR_DIR = "/Users/korth/Projects/dynamic-scene-graphs/data/living_room_1/color"
DEPTH_DIR = "/Users/korth/Projects/dynamic-scene-graphs/data/living_room_1/depth"

def load_all_color_images() -> List[np.ndarray]:
    """Load all color images from the color directory as numpy arrays."""
    color_files = sorted([f for f in os.listdir(COLOR_DIR) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(COLOR_DIR, fname), cv2.IMREAD_COLOR) for fname in color_files]
    return images

def load_all_depth_images() -> List[np.ndarray]:
    """Load all depth images from the depth directory as numpy arrays."""
    depth_files = sorted([f for f in os.listdir(DEPTH_DIR) if f.endswith('.png')])
    images = [cv2.imread(os.path.join(DEPTH_DIR, fname), cv2.IMREAD_UNCHANGED) for fname in depth_files]
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
