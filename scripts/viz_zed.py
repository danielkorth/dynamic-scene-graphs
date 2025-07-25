import cv2
import numpy as np
import open3d as o3d
import os
import configparser

from utils.data_loading import load_poses
from utils.open3d_utils import create_camera_mesh


def visualize_zed_data(data_folder, pose_file, cam_params_file, max_frames=10, subsample=1):
    # Load camera intrinsics
    config = configparser.ConfigParser()
    config.read(cam_params_file)
    section = 'LEFT_CAM_HD'
    fx = float(config[section]['fx'])
    fy = float(config[section]['fy'])
    cx = float(config[section]['cx'])
    cy = float(config[section]['cy'])
    
    # Create Open3D intrinsic object (assume 2K resolution)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(2208, 1242, fx, fy, cx, cy)
    
    # Load camera poses
    translations, rotations = load_poses(
        pose_file,
        max_frames,
        subsample
    )
    
    global_pcd = o3d.geometry.PointCloud()
    frustums = []

    n = len(translations)
    if n > 1:
        t = np.linspace(0, 1, n)
    else:
        t = np.array([0.0])
    # Create gradient: green -> yellow -> red
    colors = np.zeros((n, 3))
    mid = n // 2
    colors[:mid, 0] = np.linspace(0, 1, mid, endpoint=False)  # R: 0->1
    colors[:mid, 1] = 1                                      # G: 1
    colors[mid:, 0] = 1                                      # R: 1
    colors[mid:, 1] = np.linspace(1, 0, n - mid)             # G: 1->0
    # B stays at 0

    for i, (t_vec, rvec) in enumerate(zip(translations, rotations)):
        # Generate file paths
        frame_idx = i * subsample
        depth_path = os.path.join(data_folder, f"depth{frame_idx:06d}.png")
        color_path = os.path.join(data_folder, f"left{frame_idx:06d}.png")
        
        if not os.path.exists(depth_path) or not os.path.exists(color_path):
            print(f"Missing frame {frame_idx}, skipping")
            continue

        # Read and process depth image (convert mm to meters)
        depth_img = o3d.io.read_image(depth_path)
        depth_np = np.asarray(depth_img).astype(np.float32) / 1000.0
        
        # Create RGBD image
        bgr = cv2.imread(color_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            o3d.geometry.Image(depth_np),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic
        )
        
        # Create transformation matrix
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t_vec
        
        # Transform point cloud to world coordinates
        pcd.transform(T)
        
        # Create camera frustum with gradient color
        frustum = create_camera_mesh(scale=0.05, color=colors[i])
        frustum.transform(T)
        
        # Aggregate results
        global_pcd += pcd
        frustums.append(frustum)
    
    # Visualize
    # Add a reference frame at the origin
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([global_pcd] + frustums + [origin_frame])

# Example usage
if __name__ == "__main__":
    visualize_zed_data(
        data_folder='./data/zed/office1/images_undistorted_crop',
        pose_file='./data/zed/office1/poses.txt',
        cam_params_file='./data/SN35693142.conf',
        max_frames=None,       # Load first 50 poses
        subsample=40          # Process every 5th frame
    )