from pathlib import Path
import argparse
import rerun as rr
from utils.colmap_utils import load_colmap_poses
from utils.data_loading import load_poses, load_camera_intrinsics, load_all_rgb_images, load_all_depth_images, load_all_masks, get_camera_matrix, get_distortion_coeffs
from utils.rerun import setup_blueprint
from utils.tools import get_color_for_id
import numpy as np
from scipy.spatial.transform import Rotation
from utils.cv2_utils import unproject_image
import time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize ZED camera data in Rerun")
    parser.add_argument("--max-frames", "-n", type=int, default=None, 
                       help="Maximum number of frames to load (default: all frames)")
    parser.add_argument("--subsample", "-s", type=int, default=None,
                       help="Subsample every Nth frame (default: no subsampling)")
    parser.add_argument("--data-dir", "-d", type=str, 
                       default="./data/zed/cooking",
                       help="Path to data directory (default: data/zed/short)")
    parser.add_argument("--masks-dir", "-m", type=str,
                       default="./data/results/cooking/masks",
                       help="Path to masks directory (default: data/results/cooking/masks)")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (default: False)")
    parser.add_argument("--use_depth", action="store_false",
                       help="Do not use depth images (default: use depth images)")
    parser.set_defaults(no_depth=False)
    args = parser.parse_args()

    print(args)
    # set up rerun env
    if args.headless:
        # ensure that a rerun server is running (rerun --serve)
        # rr.init("living_room", spawn=False, default_blueprint=setup_blueprint())
        # rr.connect_grpc()
        pass
    else:
        rr.init("living_room", spawn=True, default_blueprint=setup_blueprint())

    rr.set_time(timeline="world", sequence=0)
    rr.log("world", rr.ViewCoordinates.RDF)

    intrinsics = load_camera_intrinsics("./data/SN35693142.conf", camera="left", resolution="HD")
    K = get_camera_matrix(intrinsics)
    dist = get_distortion_coeffs(intrinsics)

    data_dir = Path(args.data_dir)
    rgb_images = load_all_rgb_images(data_dir / "images", max_frames=args.max_frames, subsample=args.subsample)
    depth_images = load_all_depth_images(data_dir / "images", max_frames=args.max_frames, subsample=args.subsample)
    mask_images = load_all_masks(args.masks_dir, max_frames=args.max_frames, subsample=args.subsample)

    # zed poses
    tvecs, rvecs = load_poses(data_dir / "poses.txt", max_frames=args.max_frames, subsample=args.subsample)
    # colmap poses
    # rvecs, tvecs, _ = load_colmap_poses(data_dir / "colmap_poses.txt", max_frames=args.max_frames, subsample=args.subsample)
    # tvecs = tvecs * 0.005

    print(f"Loaded {len(rgb_images)} RGB images and {len(depth_images)} depth images")
    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    rr.log("camera", rr.ViewCoordinates.RDF)

    rr.log("camera/image", rr.Pinhole(
        resolution=[rgb_images[0].shape[1], rgb_images[0].shape[0]],
        principal_point=[intrinsics["cx"], intrinsics["cy"]],
        focal_length=[intrinsics["fx"], intrinsics["fy"]],
    ), static=True)

    # all_3d_points = []
    # centers = []
    from scenegraph.graph import SceneGraph
    from scenegraph.node import Node
    graph = SceneGraph()

    for i, (rgb, depth, tvec, rvec, mask_dict) in enumerate(zip(rgb_images, depth_images, tvecs, rvecs, mask_images)):
        rr.log("camera", rr.Transform3D(
            mat3x3=Rotation.from_rotvec(rvec).as_matrix(),
            translation=tvec,
        ))
        rr.log("camera/image/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))

        for obj_id, mask in mask_dict.items():
            if mask.sum() == 0:
                continue
            points_3d, _ = unproject_image(depth, K, dist, rvec, tvec, mask)
            centroid = np.mean(points_3d, axis=0)

            if f"obj_{obj_id}" not in graph:
                # Generate unique color for this object
                color = np.array(get_color_for_id(obj_id))
                graph.add_node(Node(f"obj_{obj_id}", centroid, color=color, pct=points_3d))
            else:
                graph.nodes[f'obj_{obj_id}'].pct = points_3d
                graph.nodes[f'obj_{obj_id}'].centroid = centroid

        print("Graph size: ", len(graph))

        graph.log_rerun(show_pct=True)

        rr.set_time(timeline="world", sequence=i)

if __name__ == "__main__":
    main()
