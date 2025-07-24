from utils.sam2_utils import (create_overlapping_subsets, detect_with_furthest, mask_first_frame, 
                                save_sam_cv2, save_points_image_cv2_obj_id, make_video_from_visualizations)
from sam2.build_sam import build_sam2_video_predictor
import hydra
import os
from PIL import Image
import numpy as np

@hydra.main(config_path="../configs", config_name="sam2_reinit.yaml")
def main(cfg):
    
    # load and subsample images
    subsets = create_overlapping_subsets(cfg.source_folder, cfg.output_folder, cfg.chunk_size, cfg.overlap, cfg.subsample)

    # load images
    predictor = build_sam2_video_predictor(
        config_file = cfg.sam.model_cfg,
        ckpt_path = cfg.sam.sam2_checkpoint,
        device = "cuda"
    )

    segmenter = hydra.utils.instantiate(cfg.sam)

    # load first image
    first_image_path = subsets[0] + "/000000.jpg"
    img = Image.open(first_image_path)
    img_np = np.array(img)
    
    masks = mask_first_frame(img_np, segmenter, viz=True)

    # inital points and labels 
    # dict: obj_id -> points, labels
    # defaultdict: obj_id -> points, labels
    from collections import defaultdict
    obj_points = defaultdict(lambda: {'points': [], 'labels': [], 'mask': None})
    for i, mask in enumerate(masks):
        points = mask['point_coords']
        labels = np.ones(len(points), dtype=np.int32)
        obj_points[i]['points'] = points
        obj_points[i]['labels'] = labels
        obj_points[i]['mask'] = mask['segmentation'].squeeze()

    # save the points image
    save_points_image_cv2_obj_id(os.path.join(subsets[0], "000000.jpg"), obj_points, os.path.join(cfg.output_folder, "frame_0_obj_id.png"))

    # LOOP THE SUBSETS
    for i in range(len(subsets)):
        # initialize tracking
        inference_state = predictor.init_state(video_path=subsets[i])

        for obj_id, obj_data in obj_points.items():
            # if we have mask, transfer it. otherwise, use points.
            if obj_data['mask'] is not None:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=obj_data['mask']
                )
                continue
            # skip empty
            elif len(obj_data['points']) > 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=obj_data['points'],
                    labels=obj_data['labels'],
                )
            else:
                print(f"No points or mask for obj {obj_id}")
                continue

        # propagate the tracking
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # save the results
        output_masks_dir = os.path.join(cfg.output_folder, f"masks_{i}")
        output_vis_dir = os.path.join(cfg.output_folder, f"visualizations_{i}")
        save_sam_cv2(video_segments, subsets[i], output_masks_dir, output_vis_dir)

        # update the points and labels for next episode
        full_mask = np.zeros_like(video_segments[len(video_segments) - 1][0])
        for obj_id, mask in video_segments[len(video_segments) - 1].items():
            obj_points[obj_id]['mask'] = mask.squeeze()
            full_mask += mask

        # Detect new regions
        new_regions = hydra.utils.instantiate(cfg.new_objects_fct)(full_mask)
        # add new categories
        next_obj_id = len(obj_points)
        for j, new_region in enumerate(new_regions):
            new_obj_id = next_obj_id + j
            obj_points[new_obj_id]['points'] = new_region['points'].astype(np.float32).reshape(-1, 2)
            obj_points[new_obj_id]['labels'] = new_region['labels']

        save_points_image_cv2_obj_id(os.path.join(subsets[i], f"{len(video_segments)-1:06d}.jpg"), obj_points, os.path.join(cfg.output_folder, f"frame_{i}_obj_id_new.png"))

    # At the end of main, after all processing:
    video_fps = getattr(cfg, 'video_fps', 15)
    make_video_from_visualizations(cfg.output_folder, video_filename="final_video.mp4", fps=video_fps)


if __name__ == "__main__":
    main()