from pathlib import Path
import argparse
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import time
import hydra
from omegaconf import DictConfig
from utils.colmap_utils import load_colmap_poses
from utils.data_loading import load_poses, load_camera_intrinsics, load_all_rgb_images, load_all_depth_images, load_all_masks, get_camera_matrix, get_distortion_coeffs, load_all_points
from utils.tools import get_color_for_id
from utils.cv2_utils import unproject_image
from scenegraph.graph import SceneGraph
from scenegraph.node import Node

def aggregate_masks(obj_points):
    mask = -1 * np.ones(obj_points[0]['mask'].shape)
    for id, obj_point in obj_points.items():
        mask[obj_point['mask']] = id
    return mask

def create_camera_geometry(tvec, rvec, scale=0.1):
    """Create a camera geometry for visualization"""
    # Create a simple camera frame
    camera_points = np.array([
        [0, 0, 0],      # camera center
        [scale, scale, scale],   # top-right
        [scale, -scale, scale],  # bottom-right
        [-scale, -scale, scale], # bottom-left
        [-scale, scale, scale],  # top-left
    ])
    
    # Transform points by rotation and translation
    R = Rotation.from_rotvec(rvec).as_matrix()
    camera_points = camera_points @ R.T + tvec
    
    # Create lines for camera frame
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # from center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # connecting corners
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(camera_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # Red camera frame
    
    return line_set

def create_trajectory_geometry(tvecs):
    """Create trajectory line geometry"""
    if len(tvecs) < 2:
        return None
    
    points = np.array(tvecs)
    lines = [[i, i+1] for i in range(len(points)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # Green trajectory
    
    return line_set



@hydra.main(config_path="../configs", config_name="sam2_reinit")
def main(cfg: DictConfig):
    # Initialize Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window("Dynamic Scene Graph Visualization", width=1200, height=800)
    
    # Set up coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coordinate_frame)
    
    # Store all geometries
    geometries = {
        'objects': [],
        'cameras': [],
        'trajectory': None
    }
    
    # Load intrinsics
    with open(cfg.intrinsics_file, "r") as f:
        intrinsics = f.readlines()
        intrinsics = {
            "fx": float(intrinsics[0].split(" ")[0]),
            "fy": float(intrinsics[1].split(" ")[0]),
            "cx": float(intrinsics[2].split(" ")[0]),
            "cy": float(intrinsics[3].split(" ")[0]),
        }
    K = get_camera_matrix(intrinsics)

    # Load data
    from utils.data_loading import load_everything
    data = load_everything(cfg.images_folder, cfg.obj_points_dir, max_frames=cfg.max_frames, subsample=cfg.subsample)

    rgb_images = data["rgb"]
    depth_images = data["depth"]
    obj_points = data["obj_points"]

    # Load poses
    tvecs, rvecs = load_poses(Path(cfg.source_folder) / "poses.txt", max_frames=cfg.max_frames, subsample=cfg.subsample)

    print(f"Loaded {len(rgb_images)} RGB images and {len(depth_images)} depth images")
    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    # Initialize scene graph
    graph = SceneGraph()
    
    # Create trajectory geometry
    trajectory_geom = create_trajectory_geometry(tvecs)
    if trajectory_geom:
        geometries['trajectory'] = trajectory_geom
        vis.add_geometry(trajectory_geom, False)

    for i, (rgb, depth, tvec, rvec, obj_points) in enumerate(zip(rgb_images, depth_images, tvecs, rvecs, obj_points)):
        print(f"Processing frame {i+1}/{len(rgb_images)}")
        
        # Create camera geometry for current frame
        camera_geom = create_camera_geometry(tvec, rvec)
        geometries['cameras'].append(camera_geom)
        vis.add_geometry(camera_geom, False)
        
        # Process object points
        for obj_id, obj_point in obj_points.items():
            if obj_point['mask'].sum() == 0:
                continue
            points_3d, _ = unproject_image(depth, K, -rvec, tvec, mask=obj_point['mask'], dist=None)
            centroid = np.mean(points_3d, axis=0)

            if f"obj_{obj_id}" not in graph:
                # Generate unique color for this object
                color = np.array(get_color_for_id(obj_id))
                graph.add_node(Node(f"obj_{obj_id}", centroid, color=color, pct=points_3d))
            else:
                if cfg.accumulate_points:
                    graph.nodes[f'obj_{obj_id}'].pct = np.vstack([graph.nodes[f'obj_{obj_id}'].pct, points_3d])
                else:
                    graph.nodes[f'obj_{obj_id}'].pct = points_3d
                graph.nodes[f'obj_{obj_id}'].centroid = centroid

        print(f"Graph size: {len(graph)}")

        # Update visualization
        graph.log_open3d(vis, geometries, show_pct=True)
        
        # Update trajectory
        if i > 0 and trajectory_geom:
            # Update trajectory points
            trajectory_points = np.array(tvecs[:i+1])
            trajectory_geom.points = o3d.utility.Vector3dVector(trajectory_points)
            vis.update_geometry(trajectory_geom)
        
        # Update visualization
        vis.poll_events()
        vis.update_renderer()
        
        # Add small delay for visualization
        time.sleep(0.1)
        
        # Check if window is closed
        if not vis.poll_events():
            break

    print("Visualization complete. Press 'q' to exit.")
    
    # Keep window open until user closes it
    while vis.poll_events():
        vis.update_renderer()
    
    vis.destroy_window()

if __name__ == "__main__":
    main() 