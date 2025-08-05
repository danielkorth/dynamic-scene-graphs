from pathlib import Path
import argparse
import rerun as rr
from utils.colmap_utils import load_colmap_poses
from utils.data_loading import load_poses, load_camera_intrinsics, load_all_rgb_images, load_all_depth_images, load_all_masks, get_camera_matrix, get_distortion_coeffs, load_all_points
from utils.rerun import setup_blueprint
import numpy as np
from scipy.spatial.transform import Rotation
import time
import hydra
from omegaconf import DictConfig
from scenegraph.graph import SceneGraph, process_frame_with_representation

def aggregate_masks(obj_points):
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

    from utils.data_loading import load_everything
    data = load_everything(cfg.images_folder, cfg.obj_points_dir, max_frames=cfg.max_frames, subsample=cfg.subsample)

    rgb_images = data["rgb"]
    depth_images = data["depth"]
    obj_points = data["obj_points"]

    # zed poses
    tvecs, rvecs = load_poses(Path(cfg.source_folder) / "poses.txt", max_frames=cfg.max_frames, subsample=cfg.subsample)

    print(f"Loaded {len(rgb_images)} RGB images and {len(depth_images)} depth images")
    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    # rr.log("/", rr.AnnotationContext([  
    #     rr.AnnotationInfo(id=1, label="red", color=rr.Rgba32([255, 0, 0, 255])),  
    #     rr.AnnotationInfo(id=2, label="green", color=rr.Rgba32([0, 255, 0, 255]))  
    # ]), static=True)

#     rr.log(
#     "masks",  # Applies to all entities below "masks".
#     rr.AnnotationContext(
#         [
#             rr.AnnotationInfo(id=0, label="Background"),
#             rr.AnnotationInfo(id=1, label="Person", color=(255, 0, 0, 0)),
#         ],
#     ),
#     static=True,
# )
    # rr.log("/", rr.AnnotationContext([(1, "red", (255, 0, 0)), (2, "green", (0, 255, 0))]), static=True)

    rr.log("world/camera", rr.ViewCoordinates.RDF)

    rr.log("world/camera/image", rr.Pinhole(
        resolution=[rgb_images[0].shape[1], rgb_images[0].shape[0]],
        principal_point=[intrinsics["cx"], intrinsics["cy"]],
        focal_length=[intrinsics["fx"], intrinsics["fy"]],
    ), static=True)

    # all_3d_points = []
    # centers = []
    graph = SceneGraph()

    line_strips = []

    # Check if we should use TSDF representation
    use_tsdf = getattr(cfg, 'use_tsdf', False)
    print(f"Using TSDF representation: {use_tsdf}")

    for i, (rgb, depth, tvec, rvec, obj_points) in enumerate(zip(rgb_images, depth_images, tvecs, rvecs, obj_points)):
        rr.log("world/camera", rr.Transform3D(
            mat3x3=Rotation.from_rotvec(rvec).as_matrix(),
            translation=tvec,
        ))
        rr.log("world/camera/image/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))
        rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1000.0, depth_range=[0, 5000]))
        rr.log("world/camera/image/mask", rr.SegmentationImage(aggregate_masks(obj_points)))

        # Process frame with chosen representation
        process_frame_with_representation(rgb, depth, tvec, rvec, obj_points, K, graph, cfg, use_tsdf)

        print("Graph size: ", len(graph))

        graph.log_rerun(show_pct=True)

        # LOG CAMERA TRAJECTORY
        if i > 0:
            line_strips.append(np.array([tvecs[i-1], tvecs[i]]))
            rr.log("world/trajectory", rr.LineStrips3D(
                strips=np.array(line_strips),
                colors=np.array([[255, 0, 0, 255]] * len(line_strips)),
            ))

        rr.set_time(timeline="world", sequence=i)


if __name__ == "__main__":
    main()
