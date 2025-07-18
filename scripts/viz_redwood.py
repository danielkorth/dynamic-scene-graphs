import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
from tqdm import tqdm

import open3d as o3d
import numpy as np
import cv2
import copy

from utils.open3d_utils import (vectors_to_mat, mat_to_vectors, pc_from_rgbd, merge_pointclouds, create_camera_mesh)
from utils.redwood import read_trajectory
from utils.data_loading import (load_poses, load_camera_intrinsics, 
    get_camera_matrix, get_distortion_coeffs, load_depth_image, load_colmap_poses)
from utils.cv2_utils import unproject_image

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    # dataset = "redwood"
    dataset = "zed"

    # Example for reading the image files in a folder
    if dataset == "redwood":
        path_rgb = './data/living_room_1/livingroom1-color'
        path_depth = './data/living_room_1/livingroom1-depth-clean'
        path_traj = './data/living_room_1/livingroom1-traj.txt'

        rgdfiles = sorted(os.listdir(path_rgb))
        depthfiles = sorted(os.listdir(path_depth))

        rgdfiles = [os.path.join(path_rgb, rgdfiles[i]) for i in range(len(rgdfiles))]
        depthfiles = [os.path.join(path_depth, depthfiles[i]) for i in range(len(depthfiles))]

        traj = read_trajectory(path_traj)
        traj = [t.pose for t in traj]

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    elif dataset == "zed":
        path_images = './data/zed/short/images'
        path_traj = './data/zed/short/poses.txt'
        intrinsics_path = './data/SN35693142.conf'
        crop_px = 40

        all_files = sorted(os.listdir(path_images))
        rgdfiles = [os.path.join(path_images, f) for f in all_files if f.startswith("left")]
        depthfiles = [os.path.join(path_images, f) for f in all_files if f.startswith("depth")]

        # rot, trans = load_poses(path_traj)
        if "colmap" in path_traj:
            rot, trans, _ = load_colmap_poses(path_traj)
        else:
            rot, trans = load_poses(path_traj)

            # Flip rot and trans vectors
            def swap_xy(vec):
                vec = np.asarray(vec)
                return np.array([-vec[1], vec[0], vec[2]])

            rot = [swap_xy(r) for r in rot]
            trans = [swap_xy(t) for t in trans]

        traj = [vectors_to_mat(-r, t) for r, t in zip(rot, trans)]

        intrinsics_dict = load_camera_intrinsics(intrinsics_path)

        # Build original camera matrix and distortion coefficients
        K = get_camera_matrix(intrinsics_dict)
        dist = get_distortion_coeffs(intrinsics_dict)

    assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    examples = 20
    separation = 20
    ##########################################################################################################

    examples = len(traj) if examples==-1 else examples
    
    pointclouds = []
    
    for i in tqdm(range(0, min(examples*separation, len(traj)), separation)): #range(len(rgdfiles)):
        if dataset == "redwood":
            print(rgdfiles[i])
            pointcloud = pc_from_rgbd(rgdfiles[i], 
                                    depthfiles[i],
                                    traj[i], intrinsics, crop = crop_px)
        else:
            rvec, tvec = rot[i], trans[i]
            pointcloud_np, pixel_coords = unproject_image(load_depth_image(path_images, frame_number=i), K, dist, -rvec, tvec)
            # Load the corresponding RGB image
            rgb_img = cv2.imread(rgdfiles[i])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR

            # Ensure pixel_coords are valid integer indices
            u = np.clip(pixel_coords[:, 0].astype(int), 0, rgb_img.shape[1] - 1)
            v = np.clip(pixel_coords[:, 1].astype(int), 0, rgb_img.shape[0] - 1)

            # Retrieve colors for each point
            colors = rgb_img[v, u] / 255.0  # Normalize to [0, 1] for Open3D

            pointcloud_o3d = o3d.geometry.PointCloud()
            pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud_np)
            pointcloud_o3d.colors = o3d.utility.Vector3dVector(colors)
            pointclouds.append(pointcloud_o3d)
            break

    final_pc = merge_pointclouds(pointclouds, voxel_th=0.001)

    # Extract camera positions from poses
    positions = [traj[i][:3, 3] for i in range(0, min(examples*separation, len(traj)), separation)]

    # Create Open3D PointCloud from positions
    trajectory_pc = o3d.geometry.PointCloud()
    positions_np = np.array(positions)
    trajectory_pc.points = o3d.utility.Vector3dVector(positions_np)
    
    n = len(positions_np)
    if n > 1:
        t = np.linspace(0, 1, n)
    else:
        t = np.array([0.0])

    # Create gradient: green -> yellow -> red
    colors = np.zeros((n, 3))
    mid = n // 2
    # First half: green to yellow
    colors[:mid, 0] = np.linspace(0, 1, mid, endpoint=False)  # R: 0->1
    colors[:mid, 1] = 1                                      # G: 1
    # Second half: yellow to red
    colors[mid:, 0] = 1                                      # R: 1
    colors[mid:, 1] = np.linspace(1, 0, n - mid)             # G: 1->0
    # B stays at 0

    # Add a reference frame at the origin
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # Create camera frustums for each pose, but only every N steps
    camera_frustums = []
    N = 1  # Show one frustum every N steps
    for idx in range(0, examples):
        i = idx * separation
        if i % N != 0:
            continue
        frustum = create_camera_mesh(scale=0.05, color=colors[idx])
        # Apply pose (rotation and translation)
        pose = traj[i]
        frustum = copy.deepcopy(frustum)
        frustum.transform(pose)
        camera_frustums.append(frustum)

    o3d.visualization.draw_geometries(
        [final_pc, trajectory_pc, origin_frame] + camera_frustums,
        window_name="Open3D Point Cloud Viewer",
        width=800,
        height=600,
        point_show_normal=False
    )
