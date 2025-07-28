from utils.sam2_utils import (create_overlapping_subsets, detect_with_furthest, is_new_obj, mask_first_frame, 
                                save_sam_cv2, save_points_image_cv2_obj_id, make_video_from_visualizations, detect_with_cc, get_mask_from_points, save_obj_points)
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment import SAM2Segmenter
import hydra
import os
from PIL import Image
import numpy as np

@hydra.main(config_path="../configs", config_name="sam2_reinit.yaml")
def main(cfg):
    # load and subsample images
    subsets = create_overlapping_subsets(cfg.images_folder, cfg.output_folder, cfg.chunk_size, cfg.overlap, cfg.subsample)

    obj_points_dir = os.path.join(cfg.output_folder, "obj_points_history")
    os.makedirs(obj_points_dir, exist_ok=True)

    # load images
    predictor = build_sam2_video_predictor(
        config_file = cfg.sam.model_cfg,
        ckpt_path = cfg.sam.sam2_checkpoint,
        device = "cuda"
    )

    auto_segmenter = hydra.utils.instantiate(cfg.sam)
    img_segmenter = SAM2ImagePredictor(auto_segmenter.sam2)

    # load first image
    first_image_path = subsets[0] + "/000000.jpg"
    img = Image.open(first_image_path)
    img_np = np.array(img)
    
    masks = mask_first_frame(img_np, auto_segmenter, viz=True)

    ### TESTING SPACE
    # accumulate masks
    full_mask = np.zeros_like(masks[0]['segmentation'])
    for mask in masks:
        full_mask += mask['segmentation']

    # detect new regions
    new_regions = hydra.utils.instantiate(cfg.new_objects_fct)(full_mask, mask_generator=auto_segmenter.mask_generator, image=img_np, viz=True)

    ### END TESTING SPACE

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
    save_obj_points(obj_points, os.path.join(obj_points_dir, "obj_points_0.npy"))

    global_counter = 1
    # LOOP THE SUBSETS
    for i in range(len(subsets)):
        # initialize tracking
        inference_state = predictor.init_state(video_path=subsets[i])

        # reinit the inference state and add new points and labels
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
        for j, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(predictor.propagate_in_video(inference_state)):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            # update mask
            for obj_id, mask in video_segments[out_frame_idx].items():
                obj_points[obj_id]['mask'] = mask.squeeze()
            # save item
            if j > 0: # skip first because of overlap
                save_obj_points(obj_points, os.path.join(obj_points_dir, f"obj_points_{global_counter*cfg.subsample}.npy"))
                global_counter += 1

        # save the results
        output_masks_dir = os.path.join(cfg.output_folder, f"masks_{i}")
        output_vis_dir = os.path.join(cfg.output_folder, f"visualizations_{i}")
        save_sam_cv2(video_segments, subsets[i], output_masks_dir, output_vis_dir)
    
        # update the points and labels for next episode
        full_mask = np.zeros_like(video_segments[len(video_segments) - 1][0])
        for obj_id, mask in video_segments[len(video_segments) - 1].items():
            full_mask += mask

        last_img_patch_path = subsets[i] + f"/{len(video_segments)-1:06d}.jpg"
        img_last_patch = Image.open(last_img_patch_path)
        img_last_patch_np = np.array(img_last_patch)

        # Detect new regions
        # new_regions = hydra.utils.instantiate(cfg.new_objects_fct)(full_mask)
        new_regions = hydra.utils.instantiate(cfg.new_objects_fct)(full_mask, mask_generator=auto_segmenter.mask_generator, image=img_last_patch_np)

        if len(new_regions) > 0:
            if cfg.prompt_with_masks:
                all_points = np.vstack([new_regions[i]['points'] for i in range(len(new_regions))])
                valids_idx = get_mask_from_points(all_points, img_segmenter, img_last_patch_np, iou_threshold=0.5)
                new_regions = [new_regions[i] for i in valids_idx]

            # add new categories
            next_obj_id = max(obj_points.keys()) + 1
            for j, new_region in enumerate(new_regions):
                if new_region['mask'] is None or is_new_obj(new_region['mask'], obj_points, iou_threshold=0.5):
                    new_obj_id = next_obj_id + j
                    obj_points[new_obj_id]['points'] = new_region['points'].astype(np.float32).reshape(-1, 2) if new_region['points'] is not None else None
                    obj_points[new_obj_id]['labels'] = new_region['labels'] if new_region['labels'] is not None else None
                    obj_points[new_obj_id]['mask'] = new_region['mask']
                else:
                    print(f"New region {j} is not new")

        # save_points_image_cv2_obj_id(os.path.join(subsets[i], f"{len(video_segments)-1:06d}.jpg"), obj_points, os.path.join(cfg.output_folder, f"frame_{i}_obj_id_new.png"))

    # At the end of main, after all processing:
    video_fps = getattr(cfg, 'video_fps', 15)
    make_video_from_visualizations(cfg.output_folder, video_filename="final_video.mp4", fps=video_fps)


if __name__ == "__main__":
    main()