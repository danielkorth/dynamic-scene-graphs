import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.open3d_utils import read_trajectory, pc_from_rgbd_with_mask

# --------------------------------------------
# Utility: generate a consistent color per object ID
# --------------------------------------------
def get_color_for_id(obj_id):
    # Golden angle in degrees
    golden_angle = 137.508
    hue = (obj_id * golden_angle) % 360 / 360.0
    # Convert HSV to RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.90)
    return [r, g, b]

# --------------------------------------------
# Main
# --------------------------------------------
if __name__ == '__main__':
    # Paths
    path_rgb   = './data/living_room_1/livingroom1-color'
    path_depth = './data/living_room_1/livingroom1-depth-clean'
    traj_file  = './data/living_room_1/livingroom1-traj.txt'
    mask_dir   = './data/sam2_res/masks'
    vis_dir    = './data/sam2_res/visualizations'

    # Read trajectory
    traj = read_trajectory(traj_file)

    # Determine which frames have SAM masks (via visualizations)
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
    frame_ids = [os.path.splitext(f)[0] for f in vis_files]

    all_pcs = []
    seen_ids = set()

    for fid in tqdm(frame_ids, desc="Processing SAM frames"):
        # Locate RGB & depth for this frame
        # Assumes files named like fid + ext (.png/.jpg)
        fid_rgbd = '0' + fid[5:]
        col_path = os.path.join(path_rgb, fid_rgbd + '.png')
        dep_path = os.path.join(path_depth, fid_rgbd + '.png')
        # Fallback to .jpg if needed
        if not os.path.exists(col_path):
            col_path = os.path.join(path_rgb, fid_rgbd + '.jpg')
        if not os.path.exists(dep_path):
            dep_path = os.path.join(path_depth, fid_rgbd + '.jpg')

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

            # Backproject
            pcd = pc_from_rgbd_with_mask(col_path, dep_path, mask, traj[int(fid.replace('frame', ''))].pose)

            # Color uniformly
            color = get_color_for_id(obj_id)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))

            all_pcs.append(pcd)

    # Merge all into one cloud and downsample
    merged = o3d.geometry.PointCloud()
    for pc in all_pcs:
        merged += pc
    merged = merged.voxel_down_sample(voxel_size=0.01)

    # Visualize
    o3d.visualization.draw_geometries([merged])
