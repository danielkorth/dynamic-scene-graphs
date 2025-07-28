import os
import glob
import numpy as np
import open3d as o3d
import configparser
from tqdm import tqdm
from utils.data_loading import load_poses
from utils.open3d_utils import pc_from_rgbd_with_mask, rt_to_mat, create_camera_mesh
from utils.tools import get_color_for_id
from utils.sam2_utils import load_obj_points

if __name__ == '__main__':
    # Paths (edit as needed)
    data_folder = './data/zed/office1/images_undistorted_crop'
    pose_file = './data/zed/office1/poses.txt'
    cam_params_file = './data/zed/office1/images_undistorted_crop/intrinsics.txt'
    sam_dir = './outputs/2025-07-25/14-38-06/'  # Directory with obj_points_history
    obj_points_dir = os.path.join(sam_dir, 'obj_points_history')

    # Load camera intrinsics
    if cam_params_file.endswith('.conf'):
        config = configparser.ConfigParser()
        config.read(cam_params_file)
        section = 'LEFT_CAM_HD'
        fx = float(config[section]['fx'])
        fy = float(config[section]['fy'])
        cx = float(config[section]['cx'])
        cy = float(config[section]['cy'])
    elif cam_params_file.endswith('.txt'):
        with open(cam_params_file, "r") as f:
            intrinsics = f.readlines()
            fx = float(intrinsics[0].split(" ")[0])
            fy = float(intrinsics[1].split(" ")[0])
            cx = float(intrinsics[2].split(" ")[0])
            cy = float(intrinsics[3].split(" ")[0])
    else:
        raise ValueError(f"Unsupported camera parameters file format: {cam_params_file}")

    # Get width and height from the first color image
    sample_color_path = None
    for f in sorted(os.listdir(data_folder)):
        if f.startswith('left') and f.endswith('.png'):
            sample_color_path = os.path.join(data_folder, f)
            break
    if sample_color_path is None:
        raise RuntimeError(f"No color images found in {data_folder}")
    sample_color = o3d.io.read_image(sample_color_path)
    sample_color_np = np.asarray(sample_color)
    height, width = sample_color_np.shape[0], sample_color_np.shape[1]

    # Load camera poses
    translations, rotations = load_poses(pose_file, max_frames=None, subsample=1)

    # Find all obj_points_*.npy files
    obj_points_files = sorted(glob.glob(os.path.join(obj_points_dir, 'obj_points_*.npy')), key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))

    all_pcs = []
    frustums = []
    for obj_file in tqdm(obj_points_files, desc='Processing obj_points_history'):
        # Extract global frame index from filename
        base = os.path.basename(obj_file)
        global_idx = int(base.split('_')[2].split('.')[0])
        color_path = os.path.join(data_folder, f"left{global_idx:06d}.png")
        depth_path = os.path.join(data_folder, f"depth{global_idx:06d}.png")
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            continue
        # Load color and depth (for mask shape check)
        color = o3d.io.read_image(color_path)
        depth = o3d.io.read_image(depth_path)
        # Load obj_points dict
        obj_points = load_obj_points(obj_file)
        # For each object id, get mask and unproject
        for obj_id, data in obj_points.items():
            mask = data.get('mask', None)
            if mask is None or not np.any(mask):
                continue
            t_vec = translations[global_idx]
            rvec = rotations[global_idx]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            pcd = pc_from_rgbd_with_mask(color_path, depth_path, mask, rt_to_mat(rvec, t_vec), intrinsic)
            color_rgb = get_color_for_id(int(obj_id))
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color_rgb, (len(pcd.points), 1)))
            all_pcs.append(pcd)
        # Add frustum for this pose
        T = rt_to_mat(rotations[global_idx], translations[global_idx])
        frustum = create_camera_mesh(scale=0.05, color=[0.2, 0.8, 1.0])
        frustum.transform(T)
        frustums.append(frustum)

    # Merge all into one cloud and downsample
    merged = o3d.geometry.PointCloud()
    for pc in all_pcs:
        merged += pc
    merged = merged.voxel_down_sample(voxel_size=0.01)
    # Visualize
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([merged] + frustums + [origin_frame]) 