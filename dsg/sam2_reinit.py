from dsg.utils.sam2_utils import (create_overlapping_subsets, mask_first_frame, 
                                save_sam_cv2, make_video_from_visualizations, save_obj_points, solve_overlap, reident_new_masks)
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dsg.features.dinov2 import DINOv2
from dsg.features.dinov3 import DINOv3
from dsg.features.salad import SALAD
from dsg.features.clip_features import CLIPFeatures
import hydra
import os
from PIL import Image
import numpy as np
import supervision as sv
from supervision.detection.utils.converters import mask_to_xyxy
import cv2

@hydra.main(config_path="../configs", config_name="sam2_reinit.yaml")
def main(cfg):
    # load and subsample images
    import os
    original_cwd = hydra.utils.get_original_cwd()
    print(original_cwd)
    os.chdir(original_cwd)
    print(os.getcwd())
    subsets = create_overlapping_subsets(cfg.images_folder, cfg.output_folder, cfg.chunk_size, cfg.overlap, cfg.subsample)

    obj_points_dir = os.path.join(cfg.source_folder, "obj_points_history")
    os.makedirs(obj_points_dir, exist_ok=True)

    # Initialize DINOv2 for feature extraction
    # dinov2_extractor = DINOv2()
    salad_extractor = SALAD()
    dinov2_extractor = DINOv3()
    clip_extractor = CLIPFeatures()

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
    
    masks = mask_first_frame(img_np, auto_segmenter, viz=False)

    # create crop folder
    crop_dir = os.path.join(cfg.output_folder, "crop")
    os.makedirs(crop_dir, exist_ok=True)

    # inital points and labels 
    # dict: obj_id -> points, labels
    # defaultdict: obj_id -> points, labels
    from collections import defaultdict
    obj_points = defaultdict(lambda: {'points': [], 'labels': [], 'mask': None, 'crop': None, 'dinov2_features': None, 'salad_features': None, 'clip_features': None})
    for i, mask in enumerate(masks):
        points = mask['point_coords']
        labels = np.ones(len(points), dtype=np.int32)
        obj_points[i]['points'] = points
        obj_points[i]['labels'] = labels
        obj_points[i]['mask'] = mask['segmentation'].squeeze()

        cropped_image = sv.crop_image(img_np, mask_to_xyxy(mask['segmentation'][None, ...])) 
        crop_path = os.path.join(crop_dir, f"cropped_image_{i}.jpg")
        cv2.imwrite(crop_path, cropped_image)
        obj_points[i]['crop'] = cropped_image
        
        # Extract DINOv2 features immediately
        dinov2_features = dinov2_extractor.extract_features(crop_path)
        obj_points[i]['dinov2_features'] = dinov2_features.cpu().numpy()
        salad_features = salad_extractor.extract_features(crop_path)
        obj_points[i]['salad_features'] = salad_features.cpu().numpy()
        clip_vision_features = clip_extractor.extract_vision_features(crop_path)
        obj_points[i]['clip_features'] = clip_vision_features.cpu().numpy()

    # save the points image
    # save_points_image_cv2_obj_id(os.path.join(subsets[0], "000000.jpg"), obj_points, os.path.join(cfg.output_folder, "frame_0_obj_id.png"))
    
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

        new_regions = hydra.utils.instantiate(cfg.new_objects_fct)(full_mask, mask_generator=auto_segmenter.mask_generator, image=img_last_patch_np)

        if len(new_regions) > 0:
            # if new_regions[0]['mask'] is None and cfg.prompt_with_masks:
            #     last_img_patch_path = subsets[i] + f"/{len(video_segments)-1:06d}.jpg"
            #     img_last_patch = Image.open(last_img_patch_path)
            #     img_last_patch_np = np.array(img_last_patch)
            #     all_points = np.vstack([new_regions[i]['points'] for i in range(len(new_regions))])
            #     valids_idx, valid_masks = get_mask_from_points(all_points, img_segmenter, img_last_patch_np, iou_threshold=0.5)
            #     new_regions = [new_regions[i] for i in valids_idx]
            #     for v_i, mask in enumerate(valid_masks):
            #         new_regions[v_i]['mask'] = mask
            
            obj_points, new_regions = solve_overlap(obj_points, new_regions)

            # add new categories
            num_obj_last_it = max(obj_points.keys())
            next_obj_id = num_obj_last_it + 1
            for j, new_region in enumerate(new_regions):
                ### get rop and extract features
                new_crop = sv.crop_image(img_last_patch_np, mask_to_xyxy(new_region['mask'][None, ...]))
                crop_path = os.path.join(crop_dir, f"cropped_image_{next_obj_id}.jpg")
                cv2.imwrite(crop_path, new_crop)
                # Extract DINOv2 features immediately for new objects
                dinov2_features = dinov2_extractor.extract_features(crop_path).cpu().numpy()
                salad_features = salad_extractor.extract_features(crop_path).cpu().numpy()
                clip_vision_features = clip_extractor.extract_vision_features(crop_path).cpu().numpy()

                ### check if the new object is new
                new_obj_id = reident_new_masks(obj_points, num_obj_last_it, dinov2_features, threshold=0.4, viz=True, new_crop=new_crop, output_dir=cfg.output_folder + "/reidentification", idx1=i, idx2=j)
                if new_obj_id == -1:
                    new_obj_id = next_obj_id
                    next_obj_id += 1

                obj_points[new_obj_id]['mask'] = new_region['mask']
                obj_points[new_obj_id]['crop'] = new_crop
                obj_points[new_obj_id]['dinov2_features'] = dinov2_features
                obj_points[new_obj_id]['salad_features'] = salad_features
                obj_points[new_obj_id]['clip_features'] = clip_vision_features

    # At the end of main, after all processing:
    video_fps = getattr(cfg, 'video_fps', 15)
    make_video_from_visualizations(cfg.output_folder, video_filename="final_video.mp4", fps=video_fps)


if __name__ == "__main__":
    main()