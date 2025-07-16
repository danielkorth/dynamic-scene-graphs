from pathlib import Path
import argparse
import rerun as rr
from utils.zed import load_poses, load_camera_intrinsics, load_all_rgb_images, load_all_depth_images, get_camera_matrix, get_distortion_coeffs
from utils.rerun import setup_blueprint
import numpy as np
from scipy.spatial.transform import Rotation
from utils.cv2_utils import unproject_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize ZED camera data in Rerun")
    parser.add_argument("--max-frames", "-n", type=int, default=None, 
                       help="Maximum number of frames to load (default: all frames)")
    parser.add_argument("--subsample", "-s", type=int, default=None,
                       help="Subsample every Nth frame (default: no subsampling)")
    parser.add_argument("--data-dir", "-d", type=str, 
                       default="/local/home/dkorth/Projects/dynamic-scene-graphs/data/recent",
                       help="Path to data directory (default: data/recent)")
    args = parser.parse_args()

    # set up rerun env
    rr.init("living_room", spawn=False, default_blueprint=setup_blueprint())
    rr.connect_grpc()

    rr.set_time(timeline="world", sequence=0)

    intrinsics = load_camera_intrinsics("/local/home/dkorth/Projects/dynamic-scene-graphs/data/SN35693142.conf", camera="left", resolution="2K")
    K = get_camera_matrix(intrinsics)
    dist = get_distortion_coeffs(intrinsics)

    data_dir = Path(args.data_dir)
    rgb_images = load_all_rgb_images(data_dir / "images", max_frames=args.max_frames, subsample=args.subsample)
    depth_images = load_all_depth_images(data_dir / "images", max_frames=args.max_frames, subsample=args.subsample)
    tvecs, rvecs = load_poses(data_dir / "poses.txt", max_frames=args.max_frames, subsample=args.subsample)

    print(f"Loaded {len(rgb_images)} RGB images and {len(depth_images)} depth images")
    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    rr.log("world/camera/image", rr.ViewCoordinates.RDF)

    rr.log("world/camera/image", rr.Pinhole(
        resolution=[rgb_images[0].shape[1], rgb_images[0].shape[0]],
        principal_point=[intrinsics["cx"], intrinsics["cy"]],
        focal_length=[intrinsics["fx"], intrinsics["fy"]],
    ))

    for i, (rgb, depth, tvec, rvec) in enumerate(zip(rgb_images, depth_images, tvecs, rvecs)):
        rr.log("world/camera/image/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))
        # rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1000, depth_range=[0, 3000]))

        points_3d, _ = unproject_image(depth, K, dist, rvec, tvec)
        rr.log("world/points", rr.Points3D(points_3d, colors=[255, 0, 0]))

        rr.log("world/camera", rr.Transform3D(
            rotation=rr.RotationAxisAngle(axis=rvec, angle=-float(np.linalg.norm(rvec))),
            translation=-Rotation.from_rotvec(rvec).inv().apply(tvec)
        ))
        rr.set_time(timeline="world", sequence=i)

if __name__ == "__main__":
    main()