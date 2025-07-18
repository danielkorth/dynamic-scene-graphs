import os
import glob
import numpy as np
import open3d as o3d
import configparser
from tqdm import tqdm
from utils.data_loading import load_poses
from utils.open3d_utils import pc_from_rgbd_with_mask, rt_to_mat, create_camera_mesh
from utils.tools import get_color_for_id

# --------------------------------------------
# Main
# --------------------------------------------
if __name__ == '__main__':
    # Paths (edit as needed)
    data_folder = './data/zed/cooking/images'
    pose_file = './data/zed/cooking/poses.txt'
    cam_params_file = './data/SN35693142.conf'
    mask_dir = './data/sam2_res/zed_cooking/masks'  # Directory with frameXXXX_objY.npy
    vis_dir = './data/sam2_res/zed_cooking/visualizations'  # Used to get frame names

    # Load camera intrinsics
    config = configparser.ConfigParser()
    config.read(cam_params_file)
    section = 'LEFT_CAM_2K'
    fx = float(config[section]['fx'])
    fy = float(config[section]['fy'])
    cx = float(config[section]['cx'])
    cy = float(config[section]['cy'])
    width = 2208
    height = 1242

    # Load camera poses
    translations, rotations = load_poses(pose_file, max_frames=None, subsample=1)

    # Create frustums for each pose (as in viz_zed.py)
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
    frustums = []

    # Get frame IDs from vis_dir
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
    frame_ids = [os.path.splitext(f)[0] for f in vis_files]

    all_pcs = []
    for fid in tqdm(frame_ids, desc="Processing SAM frames"):
        # Frame index (assume format frameXXXX)
        frame_idx = int(fid.replace('frame', ''))
        # Color and depth image paths
        color_path = os.path.join(data_folder, f"left{frame_idx:06d}.png")
        depth_path = os.path.join(data_folder, f"depth{frame_idx:06d}.png")
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            continue
        # Load color and depth
        color = o3d.io.read_image(color_path)
        depth = o3d.io.read_image(depth_path)
        # Load all masks for this frame
        mask_pattern = os.path.join(mask_dir, f"{fid}_obj*.npy")
        mask_files = sorted(glob.glob(mask_pattern))
        for mfile in mask_files:
            # Parse object ID
            base = os.path.basename(mfile)
            parts = base.replace('.npy', '').split('_obj')
            obj_id = int(parts[1])
            # Load mask
            mask = np.load(mfile).astype(bool)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            elif mask.ndim > 2:
                mask = np.any(mask, axis=0)
            # Backproject only masked pixels
            # Get pose for this frame
            t_vec = translations[frame_idx]
            rvec = rotations[frame_idx]
            # Create Open3D intrinsic
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            # Use utility function if available, else do it here
            pcd = pc_from_rgbd_with_mask(color_path, depth_path, mask, rt_to_mat(rvec, t_vec), intrinsic)
            # Color uniformly
            color_rgb = get_color_for_id(obj_id)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color_rgb, (len(pcd.points), 1)))
            all_pcs.append(pcd)
        # After processing the point cloud for this frame, add the frustum for this pose
        T = rt_to_mat(rotations[frame_idx], translations[frame_idx])
        frustum = create_camera_mesh(scale=0.05, color=colors[frame_idx])
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