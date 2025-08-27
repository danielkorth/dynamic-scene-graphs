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

@hydra.main(config_path="../configs", config_name="sam2_reinit")
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

    # Clean up first frame to save memory
    del first_rgb

    from scenegraph.graph import SceneGraph
    from scenegraph.node import Node
    graph = SceneGraph()

    line_strips = []

    # Process frames one at a time with lazy loading
    for i in range(num_frames):
        # Load data for current frame on-demand
        frame_idx = i * cfg.subsample if cfg.subsample > 0 else i

        # Always load RGB, depth, and poses every frame for smooth visualization
        try:
            rgb = load_rgb_image(cfg.images_folder, frame_idx)
        except FileNotFoundError:
            print(f"Warning: Could not load RGB image for frame {frame_idx}, skipping")
            continue

        try:
            depth = load_depth_image(cfg.images_folder, frame_idx)
        except FileNotFoundError:
            print(f"Warning: Could not load depth image for frame {frame_idx}, skipping")
            continue

        # Get poses for this frame (already loaded efficiently)
        tvec = tvecs[i]
        rvec = rvecs[i]
        rr.log("world/camera", rr.Transform3D(
            mat3x3=Rotation.from_rotvec(rvec).as_matrix(),
            translation=tvec,
        ))
        rr.log("world/camera/image/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))
        rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1000.0, depth_range=[0, 5000]))

        # Only update scene graph every N frames to reduce computation
        should_update_graph = (i % cfg.graph_update_every == 0)

        if should_update_graph:
            # Load object points only when updating graph
            if cfg.obj_points_dir is not None:
                obj_points_file = os.path.join(cfg.obj_points_dir, f"obj_points_{frame_idx}.npy")
                if os.path.exists(obj_points_file):
                    obj_points = load_obj_points(obj_points_file)
                else:
                    obj_points = {}  # Empty dict if no object points file
            else:
                obj_points = {}

            # Log masks
            rr.log("world/camera/image/mask", rr.SegmentationImage(aggregate_masks(obj_points)))

            # Reset visibility for all existing nodes
            for node_name in graph.nodes:
                graph.nodes[node_name].visible = False

            for obj_id, obj_point in obj_points.items():
                is_visible = obj_point['mask'].sum() > 0

                if not is_visible:
                    continue

                points_3d, _ = unproject_image(depth, K, -rvec, tvec, mask=obj_point['mask'], dist=None)
                centroid = np.mean(points_3d, axis=0)

                if f"obj_{obj_id}" not in graph:
                    # Generate unique color for this object
                    color = np.array(get_color_for_id(obj_id))
                    node = Node(f"obj_{obj_id}", centroid, color=color, pct=points_3d)
                    node.visible = True
                    graph.add_node(node)
                else:
                    node = graph.nodes[f'obj_{obj_id}']
                    node.visible = True  # Mark as visible

                    # Only accumulate/update points for visible nodes
                    if cfg.accumulate_points:
                        node.pct = np.vstack([node.pct, points_3d])
                    else:
                        node.pct = points_3d
                    node.centroid = centroid

            print(f"Graph size: {len(graph)} (updated at frame {i})")

            # Pass current timestep for animation transitions
            # You can adjust transition_start, transition_duration, and final_edge_thickness here
            graph.log_rerun(show_pct=True, timestep=i, transition_start=900, transition_duration=200, final_edge_thickness=0.008)

            # Clean up object points after graph update
            del obj_points
        else:
            # For non-graph-update frames, use empty object points for mask logging
            rr.log("world/camera/image/mask", rr.SegmentationImage(aggregate_masks({}, default_shape=depth.shape)))

        # Log graph - only show point clouds on update frames to avoid expensive sampling
        if should_update_graph:
            # Pass current timestep for animation transitions
            graph.log_rerun(show_pct=True, timestep=i, transition_start=900, transition_duration=200, final_edge_thickness=0.015)  # Full logging with point clouds
        # else:
            # For non-update frames, skip expensive point cloud sampling but still log centroids
            # This provides smooth camera movement without expensive computation
            # for node_name in graph.nodes:
            #     node = graph.nodes[node_name]
            #     # Only log centroids, skip point cloud sampling
            #     rr.log(f"world/centroids/{node_name}", rr.Points3D(
            #         positions=node.centroid,
            #         radii=0.08,  # Match the radius used in graph.log_rerun()
            #         class_ids=np.array([node.id])
            #     ))

        # LOG CAMERA TRAJECTORY
        if i > 0:
            line_strips.append(np.array([tvecs[i-1], tvecs[i]]))
            rr.log("world/trajectory", rr.LineStrips3D(
                strips=np.array(line_strips),
                colors=np.array([[255, 0, 0, 255]] * len(line_strips)),
            ))

        rr.set_time(timeline="world", sequence=i)

        # Clean up frame data to prevent memory accumulation
        del rgb, depth


if __name__ == "__main__":
    main()
