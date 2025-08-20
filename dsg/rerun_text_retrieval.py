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

    from utils.data_loading import load_everything
    data = load_everything(cfg.images_folder, cfg.obj_points_dir, max_frames=cfg.max_frames, subsample=cfg.subsample)

    rgb_images = data["rgb"]
    depth_images = data["depth"]
    obj_points = data["obj_points"]

    # zed poses
    tvecs, rvecs = load_poses(Path(cfg.source_folder) / "poses.txt", max_frames=cfg.max_frames, subsample=cfg.subsample)

    print(f"Loaded {len(rgb_images)} RGB images and {len(depth_images)} depth images")
    print(f"Loaded {len(tvecs)} poses (translations: {len(tvecs)}, rotations: {len(rvecs)})")

    rr.log("world/camera", rr.ViewCoordinates.RDF)

    rr.log("world/camera/image", rr.Pinhole(
        resolution=[rgb_images[0].shape[1], rgb_images[0].shape[0]],
        principal_point=[intrinsics["cx"], intrinsics["cy"]],
        focal_length=[intrinsics["fx"], intrinsics["fy"]],
    ), static=True)

    # all_3d_points = []
    # centers = []
    from scenegraph.graph import SceneGraph
    from scenegraph.node import Node
    graph = SceneGraph()

    line_strips = []

    # log the points from last timestep

    for i, (rgb, depth, tvec, rvec, obj_points) in enumerate(zip(rgb_images, depth_images, tvecs, rvecs, obj_points)):
        if i == len(rgb_images) - 1:
            rr.log("world/camera", rr.Transform3D(
                mat3x3=Rotation.from_rotvec(rvec).as_matrix(),
                translation=tvec,
            ))
            rr.log("world/camera/image/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))
            rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1000.0, depth_range=[0, 5000]))
            rr.log("world/camera/image/mask", rr.SegmentationImage(aggregate_masks(obj_points)))
            graph.log_rerun(show_pct=True)
            graph.highlight_clip_feature_similarity_progressive()

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
            graph.nodes[f'obj_{obj_id}'].clip_features = obj_point['clip_features']

        # # LOG CAMERA TRAJECTORY
        # if i > 0:
        #     line_strips.append(np.array([tvecs[i-1], tvecs[i]]))
        #     rr.log("world/trajectory", rr.LineStrips3D(
        #         strips=np.array(line_strips),
        #         colors=np.array([[255, 0, 0, 255]] * len(line_strips)),
        #     ))

        rr.set_time(timeline="world", sequence=i)


if __name__ == "__main__":
    main()
