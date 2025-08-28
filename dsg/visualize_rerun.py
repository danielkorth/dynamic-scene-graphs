from pathlib import Path
import rerun as rr
from dsg.utils.data_loading import load_poses, get_camera_matrix
from dsg.utils.rerun import setup_blueprint
from dsg.utils.tools import get_color_for_id
import numpy as np
from scipy.spatial.transform import Rotation
from dsg.utils.cv2_utils import unproject_image
import hydra
from omegaconf import DictConfig
from dsg.scenegraph.graph import SceneGraph, process_frame_with_representation
import open3d as o3d

def aggregate_masks(obj_points, default_shape=None):
    if not obj_points:
        # Return empty mask with default shape or use first available mask shape
        if default_shape is not None:
            return -1 * np.ones(default_shape)
        else:
            # If no default shape and no objects, return a small default mask
            return -1 * np.ones((480, 640))  # Common camera resolution

    mask = -1 * np.ones(obj_points[0]['mask'].shape)
    for id, obj_point in obj_points.items():
        mask[obj_point['mask']] = id
    return mask

@hydra.main(config_path="../configs", config_name="video_tracking")
def main(cfg: DictConfig):
    # set up rerun env
    if cfg.headless:
        # ensure that a rerun server is running (rerun --serve)
        rr.init("living_room", spawn=False, default_blueprint=setup_blueprint())
        rr.connect_grpc()
        pass
    else:
        rr.init("living_room", spawn=True, default_blueprint=setup_blueprint())

    rr.set_time(timeline="world", sequence=0)
    rr.log("world", rr.ViewCoordinates.RDF)

    # load intrinsics
    with open(cfg.intrinsics_file, "r") as f:
        intrinsics = f.readlines()
        intrinsics = {
            "fx": float(intrinsics[0].split(" ")[0]),
            "fy": float(intrinsics[1].split(" ")[0]),
            "cx": float(intrinsics[2].split(" ")[0]),
            "cy": float(intrinsics[3].split(" ")[0]),
        }
    K = get_camera_matrix(intrinsics)

    # with distortion
    # intrinsics = load_camera_intrinsics("./data/SN35693142.conf", camera="left", resolution="HD")
    # K = get_camera_matrix(intrinsics)
    # dist = get_distortion_coeffs(intrinsics)

    from utils.data_loading import load_rgb_image, load_depth_image, load_obj_points
    import os

    # Only load poses efficiently (they're already memory-efficient)
    tvecs, rvecs = load_poses(Path(cfg.source_folder) / "poses.txt", max_frames=cfg.max_frames, subsample=cfg.subsample)

    # Get list of frames to process based on the poses
    num_frames = len(tvecs)

    print(f"Will process {num_frames} frames with lazy loading")

    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    rr.log("world/camera", rr.ViewCoordinates.RDF)

    # Load first frame to get image dimensions for camera setup
    first_frame_idx = cfg.subsample if cfg.subsample > 0 else 0
    first_rgb = load_rgb_image(cfg.images_folder, first_frame_idx)

    rr.log("world/camera/image", rr.Pinhole(
        resolution=[first_rgb.shape[1], first_rgb.shape[0]],
        principal_point=[intrinsics["cx"], intrinsics["cy"]],
        focal_length=[intrinsics["fx"], intrinsics["fy"]],
        image_plane_distance=0.25,
    ), static=True)

    graph = SceneGraph()

    line_strips = []

    # Check if we should use TSDF representation
    use_tsdf = getattr(cfg, 'use_tsdf', False)
    print(f"Using TSDF representation: {use_tsdf}")

    # Load all data at once (original approach)
    rgb_images = []
    depth_images = []
    obj_points_list = []

    for i in range(num_frames):
        frame_idx = i * cfg.subsample if cfg.subsample > 0 else i

        rgb = load_rgb_image(cfg.images_folder, frame_idx)
        depth = load_depth_image(cfg.images_folder, frame_idx)

        if cfg.obj_points_dir is not None:
            obj_points_file = os.path.join(cfg.obj_points_dir, f"obj_points_{frame_idx}.npy")
            if os.path.exists(obj_points_file):
                obj_points = load_obj_points(obj_points_file)
            else:
                obj_points = {}
        else:
            obj_points = {}

        rgb_images.append(rgb)
        depth_images.append(depth)
        obj_points_list.append(obj_points)

    # Process all frames at once
    for i, (rgb, depth, tvec, rvec, obj_points) in enumerate(zip(rgb_images, depth_images, tvecs, rvecs, obj_points_list)):
        rr.log("world/camera", rr.Transform3D(
            mat3x3=Rotation.from_rotvec(rvec).as_matrix(),
            translation=tvec,
        ))
        rr.log("world/camera/image/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))
        rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1000.0, depth_range=[0, 5000]))

        # Process frame with chosen representation
        process_frame_with_representation(rgb, depth, tvec, rvec, obj_points, K, graph, cfg, use_tsdf)

        # Log graph using original log_rerun function
        graph.log_rerun(show_pct=True)

        # LOG CAMERA TRAJECTORY
        if i > 0:
            line_strips.append(np.array([tvecs[i-1], tvecs[i]]))
            rr.log("world/trajectory", rr.LineStrips3D(
                strips=np.array(line_strips),
                colors=np.array([[255, 0, 0, 255]] * len(line_strips)),
            ))

        rr.set_time(timeline="world", sequence=i)

    # Save Open3D textured pointclouds for each node
    save_textured_pointclouds(graph, cfg.source_folder)


def save_textured_pointclouds(graph, source_folder):
    """
    Save textured point clouds for all nodes in the scene graph as PLY files.

    Args:
        graph: SceneGraph object containing nodes with point cloud data
        source_folder: Base folder path where reconstructions will be saved
    """
    print("Saving textured pointclouds for all nodes...")

    # Create output directory
    output_dir = Path(source_folder) / "final_reconstructions"
    output_dir.mkdir(exist_ok=True)

    # Save pointcloud for each node
    for node_name, node in graph.nodes.items():
        if node.pct is not None and node.rgb is not None and len(node.pct) > 0:
            # Create Open3D pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(node.pct)
            pcd.colors = o3d.utility.Vector3dVector(node.rgb)  # Normalize RGB values to [0,1]

            # Save as PLY file
            output_file = output_dir / f"{node_name}_textured.ply"
            o3d.io.write_point_cloud(str(output_file), pcd)
            print(f"Saved {node_name} pointcloud with {len(node.pct)} points to {output_file}")
        else:
            print(f"Skipping {node_name} - no valid pointcloud data")

    print(f"All pointclouds saved to {output_dir}")


if __name__ == "__main__":
    main()
