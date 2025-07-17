from utils.open3d_utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
from tqdm import tqdm
from utils.redwood import read_trajectory
from utils.zed import load_poses

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

    elif dataset == "zed":
        path_images = './data/zed/short/images'
        path_traj = './data/zed/short/poses.txt'

        all_files = sorted(os.listdir(path_images))
        rgdfiles = [os.path.join(path_images, f) for f in all_files if f.startswith("left")]
        depthfiles = [os.path.join(path_images, f) for f in all_files if f.startswith("depth")]

        rot, trans = load_poses(path_traj)

        # traj = [np.linalg.inv(vectors_to_mat(r, t)) for r, t in zip(rot, trans)]
        traj = [vectors_to_mat(r, t) for r, t in zip(rot, trans)]
        # traj = [np.eye(4) for r, t in zip(rot, trans)]

    assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    method = "all" # ransac, all, icp
    separation = 1
    visual = False
    examples = 20
    separation = 20
    ##########################################################################################################

    examples = len(traj) if examples==-1 else examples
    
    pointclouds = []
    for i in tqdm(range(0, examples*separation, separation)): #range(len(rgdfiles)):
        # print(rgdfiles[i])
        pointcloud = pc_from_rgbd(rgdfiles[i], 
                                depthfiles[i],
                                traj[i])
        
        pointclouds.append(pointcloud)
        if visual:
            draw_pointclouds(pointcloud)

    final_pc = merge_pointclouds(pointclouds, voxel_th=0.001)
    # draw_pointclouds(final_pc)

    # Extract camera positions from poses
    positions = [pose[:3, 3] for pose in traj[:examples]]

    # Create Open3D PointCloud from positions
    trajectory_pc = o3d.geometry.PointCloud()
    trajectory_pc.points = o3d.utility.Vector3dVector(np.array(positions))
    trajectory_pc.paint_uniform_color([1.0, 0.2, 0.8])  # Pink color

    o3d.visualization.draw_geometries(
        [final_pc, trajectory_pc],
        window_name="Open3D Point Cloud Viewer",
        width=800,
        height=600,
        point_show_normal=False
    )
