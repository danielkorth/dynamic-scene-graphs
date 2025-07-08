from utils.open3d_utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
from tqdm import tqdm

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
    # Example for reading the image files in a folder
    path_rgb = './data/living_room_1/livingroom1-color'
    path_depth = './data/living_room_1/livingroom1-depth-clean'
    rgdfiles = sorted(os.listdir(path_rgb))
    depthfiles = sorted(os.listdir(path_depth))

    assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    method = "all" # ransac, all, icp
    separation = 1
    visual = False
    examples = 30
    ##########################################################################################################

    all_tans_errors = []
    all_rot_errors = []

    traj = read_trajectory('./data/living_room_1/livingroom1-traj.txt')
    examples = len(traj) if examples==-1 else examples
    
    pointclouds = []
    for i in tqdm(range(examples)): #range(len(rgdfiles)):
        # print(rgdfiles[i])
        pointcloud = pc_from_rgbd(os.path.join(path_rgb, rgdfiles[i]), 
                                  os.path.join(path_depth, depthfiles[i]),
                                  traj[i].pose)
        pointclouds.append(pointcloud)
        if visual:
            draw_pointclouds(pointcloud)

    final_pc = merge_pointclouds(pointclouds, voxel_th=0.01)
    draw_pointclouds(final_pc)
