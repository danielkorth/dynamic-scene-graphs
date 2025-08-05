from pathlib import Path
import argparse
from dsg.utils.data_loading import load_poses, load_camera_intrinsics, get_camera_matrix, get_distortion_coeffs
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize ZED camera data in Rerun")
    parser.add_argument("--max-frames", "-n", type=int, default=None, 
                       help="Maximum number of frames to load (default: all frames)")
    parser.add_argument("--subsample", "-s", type=int, default=None,
                       help="Subsample every Nth frame (default: no subsampling)")
    parser.add_argument("--data-dir", "-d", type=str, 
                       default="short/",
                       help="Path to data directory (default: data/recent)")
    parser.add_argument("--image-type", "-i", type=str, default="images", choices=["images", "images_undistorted_crop"])
    parser.add_argument("--resolution", type=str, default="HD", choices=["HD", "2K", "FHD", "VGA"])
    args = parser.parse_args()

    data_dir = Path("data/zed/") / args.data_dir

    if args.image_type == "images_undistorted_crop":
        with open(data_dir / args.image_type / "camera_intrinsics_cropped.txt", "r") as f:
            fx = float(f.readline().strip())
            fy = float(f.readline().strip())
            cx = float(f.readline().strip())
            cy = float(f.readline().strip())
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([0, 0, 0, 0])
    else:
        intrinsics = load_camera_intrinsics("./data/SN35693142.conf", camera="left", resolution=args.resolution)
        K = get_camera_matrix(intrinsics)
        dist = get_distortion_coeffs(intrinsics)

    # zed poses
    tvecs, rvecs = load_poses(data_dir / "poses.txt", max_frames=args.max_frames, subsample=args.subsample)

    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(len(tvecs)):
        print(f"Processing frame {i}")
        color = o3d.io.read_image(data_dir / args.image_type / f"left{i:06d}.png")
        color = o3d.geometry.Image(np.ascontiguousarray(np.asarray(color)[:, :, :3]))
        depth = o3d.io.read_image(data_dir / args.image_type / f"depth{i:06d}.png")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
        roo = Rotation.from_rotvec(rvecs[i]).as_matrix()
        trr = roo @ -tvecs[i].reshape(3, 1)
        comb = np.hstack([roo, trr])
        lol = np.concatenate([comb, np.array([[0, 0, 0, 1]])])
        volume.integrate(
            rgbd, 
            intrinsic=o3d.camera.PinholeCameraIntrinsic(width=np.asarray(depth).shape[1], height=np.asarray(depth).shape[0], intrinsic_matrix=K),
            extrinsic=lol
        )
    
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh("mesh.ply", mesh)

if __name__ == "__main__":
    main()
