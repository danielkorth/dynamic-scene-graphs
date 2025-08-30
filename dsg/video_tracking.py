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
from typing import Tuple, List


def create_multi_receptive_field_crops(image: np.ndarray, output_dir: str, base_name: str, target_size: tuple = None, original_frame: np.ndarray = None, bbox: tuple = None) -> List[str]:
    """
    Create multiple crops with different receptive fields but same output size.
    This captures different zoom levels/context around the object using actual frame content.

    Args:
        image: Input image as numpy array (the object crop)
        output_dir: Directory to save the crops
        base_name: Base filename without extension
        target_size: Target size for all crops (width, height). If None, uses original crop size
        original_frame: Original frame image for context extraction (if None, uses white padding)
        bbox: Bounding box of object in original frame (x1, y1, x2, y2) for context extraction

    Returns:
        List of paths to the created crop files
    """
    crop_paths = []
    height, width = image.shape[:2]

    if target_size is None:
        target_size = (width, height)

    target_width, target_height = target_size

    # Define receptive field configurations
    # Each config specifies how much context to include around the original crop
    receptive_fields = [
        {"name": "context_1.5x", "scale": 1.5, "desc": "widest context"},
        {"name": "context_1.25x", "scale": 1.25, "desc": "medium context"},
        {"name": "original", "scale": 1.0, "desc": "normal crop"}
    ]

    for i, field in enumerate(receptive_fields):
        scale = field["scale"]

        if scale >= 1.0:
            # Context expansion - extract larger region from original frame if available
            if original_frame is not None and bbox is not None:
                # Extract context from original frame
                frame_height, frame_width = original_frame.shape[:2]
                x1, y1, x2, y2 = bbox

                # Calculate expanded region
                obj_width = x2 - x1
                obj_height = y2 - y1
                expand_width = int(obj_width * (scale - 1.0) / 2)
                expand_height = int(obj_height * (scale - 1.0) / 2)

                # Calculate expanded bounding box
                exp_x1 = max(0, x1 - expand_width)
                exp_y1 = max(0, y1 - expand_height)
                exp_x2 = min(frame_width, x2 + expand_width)
                exp_y2 = min(frame_height, y2 + expand_height)

                # Extract expanded region
                expanded_region = original_frame[exp_y1:exp_y2, exp_x1:exp_x2].copy()

                # Resize to target size
                resized = cv2.resize(expanded_region, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                # Fallback to white padding if original frame not available
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Create larger canvas with white padding
                larger_image = np.full((new_height, new_width, 3), 255, dtype=np.uint8)

                # Place original image in center
                y_offset = (new_height - height) // 2
                x_offset = (new_width - width) // 2
                larger_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

                # Resize to target size
                resized = cv2.resize(larger_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            # This case is kept for potential future use but not used in current config
            # Zoomed in - need smaller region from center
            crop_width = int(width * scale)
            crop_height = int(height * scale)

            # Extract center region
            x_start = (width - crop_width) // 2
            y_start = (height - crop_height) // 2
            center_crop = image[y_start:y_start+crop_height, x_start:x_start+crop_width]

            # Resize to target size
            resized = cv2.resize(center_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Save crop (convert RGB to BGR for cv2.imwrite)
        crop_filename = f"{base_name}_field_{i}_{field['name']}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(crop_path, resized_bgr)
        crop_paths.append(crop_path)

    return crop_paths


def extract_multi_receptive_field_features(salad_extractor, clip_extractor, crop_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from multiple receptive field crops and return averaged features.

    Args:
        salad_extractor: SALAD feature extractor
        clip_extractor: CLIP feature extractor
        crop_paths: List of paths to crops at different receptive fields

    Returns:
        Tuple of (averaged_salad_features, averaged_clip_features)
    """
    salad_features_list = []
    clip_features_list = []

    for crop_path in crop_paths:
        # Extract SALAD features
        salad_features = salad_extractor.extract_features(crop_path).cpu().numpy()
        salad_features_list.append(salad_features)

        # Extract CLIP features
        clip_features = clip_extractor.extract_vision_features(crop_path).cpu().numpy()
        clip_features_list.append(clip_features)

    # Average features across resolutions
    avg_salad_features = np.mean(salad_features_list, axis=0)
    avg_clip_features = np.mean(clip_features_list, axis=0)

    return avg_salad_features, avg_clip_features


def update_running_average(obj_data: dict, new_salad_features: np.ndarray, new_clip_features: np.ndarray):
    """
    Update running average features for an object.

    Args:
        obj_data: Object data dictionary containing feature fields
        new_salad_features: New SALAD features to incorporate
        new_clip_features: New CLIP features to incorporate
    """
    if 'feature_count' not in obj_data:
        obj_data['feature_count'] = 0
        obj_data['salad_running_avg'] = None
        obj_data['clip_running_avg'] = None

    obj_data['feature_count'] += 1

    if obj_data['salad_running_avg'] is None:
        obj_data['salad_running_avg'] = new_salad_features.copy()
        obj_data['clip_running_avg'] = new_clip_features.copy()
    else:
        # Running average: (old_avg * count + new) / (count + 1)
        obj_data['salad_running_avg'] = (obj_data['salad_running_avg'] * (obj_data['feature_count'] - 1) + new_salad_features) / obj_data['feature_count']
        obj_data['clip_running_avg'] = (obj_data['clip_running_avg'] * (obj_data['feature_count'] - 1) + new_clip_features) / obj_data['feature_count']


@hydra.main(config_path="../configs", config_name="video_tracking.yaml")
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

    # dinov2_extractor = DINOv2()
    salad_extractor = SALAD()
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
    obj_points = defaultdict(lambda: {
        'points': [], 'labels': [], 'mask': None, 'crop': None,
        'dinov2_features': None, 'salad_features': None, 'clip_features': None,
        'salad_running_avg': None, 'clip_running_avg': None, 'feature_count': 0
    })
    for i, mask in enumerate(masks):
        points = mask['point_coords']
        labels = np.ones(len(points), dtype=np.int32)
        obj_points[i]['points'] = points
        obj_points[i]['labels'] = labels
        obj_points[i]['mask'] = mask['segmentation'].squeeze()

        cropped_image = sv.crop_image(img_np, mask_to_xyxy(mask['segmentation'][None, ...]))

        # Check if crop is valid (not empty)
        if cropped_image is None or cropped_image.size == 0:
            print(f"Warning: Empty crop for object {i}, skipping object initialization")
            continue

        crop_path = os.path.join(crop_dir, f"cropped_image_{i}.jpg")
        # Convert RGB to BGR for cv2.imwrite
        cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(crop_path, cropped_image_bgr)
        obj_points[i]['crop'] = cropped_image

        if cfg.multi_res_crop:
            # Create multi-receptive-field crops and extract averaged features
            multi_crop_dir = os.path.join(cfg.output_folder, "multi_res_crops")
            os.makedirs(multi_crop_dir, exist_ok=True)

            # Get bounding box from mask
            bbox = mask_to_xyxy(mask['segmentation'][None, ...])[0]  # Get first (and only) bbox

            crop_paths = create_multi_receptive_field_crops(
                cropped_image, multi_crop_dir, f"obj_{i}",
                original_frame=img_np, bbox=bbox
            )
            salad_features, clip_features = extract_multi_receptive_field_features(salad_extractor, clip_extractor, crop_paths)
        else:
            # Original single-resolution feature extraction
            salad_features = salad_extractor.extract_features(crop_path).cpu().numpy()
            clip_features = clip_extractor.extract_vision_features(crop_path).cpu().numpy()

        # Store features
        obj_points[i]['salad_features'] = salad_features
        obj_points[i]['clip_features'] = clip_features

        # Initialize running average with initial features
        if cfg.acc_features:
            update_running_average(obj_points[i], salad_features, clip_features)

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
                if obj_data['mask'].sum() == 0:
                    print(f"Empty mask for obj {obj_id} in subset {i}")
                    continue
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=obj_data['mask']
                )
            # skip empty
            elif len(obj_data['points']) > 0:
                raise NotImplementedError("Not implemented")
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

        # Extract and update features for all tracked objects if temporal accumulation is enabled
        if cfg.acc_features:
            last_frame_idx = len(video_segments) - 1
            last_frame_path = subsets[i] + f"/{last_frame_idx:06d}.jpg"
            last_frame_img = Image.open(last_frame_path)
            last_frame_np = np.array(last_frame_img)

            for obj_id, mask in video_segments[last_frame_idx].items():
                if obj_id in obj_points:
                    # Get crop from the last frame of current subset
                    crop = sv.crop_image(last_frame_np, mask_to_xyxy(mask))

                    # Check if crop is valid (not empty)
                    if crop is None or crop.size == 0:
                        print(f"Warning: Empty crop for object {obj_id}, skipping feature extraction")
                        continue

                    temp_crop_path = os.path.join(crop_dir, f"temp_crop_{obj_id}.jpg")
                    # Convert RGB to BGR for cv2.imwrite
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(temp_crop_path, crop_bgr)

                    if cfg.multi_res_crop:
                        # Create multi-receptive-field crops and extract averaged features
                        multi_crop_dir = os.path.join(cfg.output_folder, "multi_res_crops")
                        os.makedirs(multi_crop_dir, exist_ok=True)

                        # Get bounding box from mask
                        temp_bbox = mask_to_xyxy(mask)[0]  # Get first (and only) bbox

                        crop_paths = create_multi_receptive_field_crops(
                            crop, multi_crop_dir, f"temp_obj_{obj_id}",
                            original_frame=last_frame_np, bbox=temp_bbox
                        )
                        new_salad_features, new_clip_features = extract_multi_receptive_field_features(salad_extractor, clip_extractor, crop_paths)
                    else:
                        # Original single-resolution feature extraction
                        new_salad_features = salad_extractor.extract_features(temp_crop_path).cpu().numpy()
                        new_clip_features = clip_extractor.extract_vision_features(temp_crop_path).cpu().numpy()

                    # Update direct features and running average
                    obj_points[obj_id]['salad_features'] = new_salad_features
                    obj_points[obj_id]['clip_features'] = new_clip_features
                    update_running_average(obj_points[obj_id], new_salad_features, new_clip_features)

                    # Clean up temp file
                    if os.path.exists(temp_crop_path):
                        os.remove(temp_crop_path)

        # save the results
        output_masks_dir = os.path.join(cfg.output_folder, f"masks_{i}")
        output_vis_dir = os.path.join(cfg.output_folder, f"visualizations_{i}")
        save_sam_cv2(video_segments, subsets[i], output_masks_dir, output_vis_dir)
    
        # update the points and labels for next episode
        full_mask = np.zeros_like(list(video_segments[len(video_segments) - 1].values())[0])
        for obj_id, mask in video_segments[len(video_segments) - 1].items():
            full_mask += mask

        last_img_patch_path = subsets[i] + f"/{len(video_segments)-1:06d}.jpg"
        img_last_patch = Image.open(last_img_patch_path)
        img_last_patch_np = np.array(img_last_patch)

        new_regions = hydra.utils.instantiate(cfg.new_objects_fct)(full_mask, mask_generator=auto_segmenter.mask_generator, image=img_last_patch_np)

        if len(new_regions) > 0:
            obj_points, new_regions = solve_overlap(obj_points, new_regions)

            # add new categories
            num_obj_last_it = max(obj_points.keys())
            next_obj_id = num_obj_last_it + 1
            for j, new_region in enumerate(new_regions):
                ### get crop and extract features
                new_crop = sv.crop_image(img_last_patch_np, mask_to_xyxy(new_region['mask'][None, ...]))

                # Check if crop is valid (not empty)
                if new_crop is None or new_crop.size == 0:
                    print(f"Warning: Empty crop for new object {next_obj_id}, skipping object")
                    continue

                crop_path = os.path.join(crop_dir, f"cropped_image_{next_obj_id}.jpg")
                # Convert RGB to BGR for cv2.imwrite
                new_crop_bgr = cv2.cvtColor(new_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(crop_path, new_crop_bgr)

                if cfg.multi_res_crop:
                    # Create multi-receptive-field crops and extract averaged features
                    multi_crop_dir = os.path.join(cfg.output_folder, "multi_res_crops")
                    os.makedirs(multi_crop_dir, exist_ok=True)

                    # Get bounding box from mask
                    new_bbox = mask_to_xyxy(new_region['mask'][None, ...])[0]  # Get first (and only) bbox

                    crop_paths = create_multi_receptive_field_crops(
                        new_crop, multi_crop_dir, f"new_obj_{next_obj_id}",
                        original_frame=img_last_patch_np, bbox=new_bbox
                    )
                    salad_features, clip_features = extract_multi_receptive_field_features(salad_extractor, clip_extractor, crop_paths)
                else:
                    # Original single-resolution feature extraction
                    salad_features = salad_extractor.extract_features(crop_path).cpu().numpy()
                    clip_features = clip_extractor.extract_vision_features(crop_path).cpu().numpy()

                ### check if the new object is new
                # Use running averages if available, otherwise use current features
                comparison_features = salad_features
                if cfg.acc_features and obj_points:
                    # For reidentification, we might want to use the most recent features
                    # rather than running averages to detect appearance changes
                    comparison_features = salad_features

                new_obj_id = reident_new_masks(obj_points, num_obj_last_it, comparison_features, threshold=cfg.reid_threshold, viz=True, new_crop=new_crop, output_dir=cfg.output_folder + "/reidentification", idx1=i, idx2=j)
                # new_obj_id = -1 # for testing
                if new_obj_id == -1 or obj_points[new_obj_id]['mask'].sum() > 0:
                    new_obj_id = next_obj_id
                    next_obj_id += 1
                
                print(f"New object id: {new_obj_id}")

                obj_points[new_obj_id]['mask'] = new_region['mask']
                obj_points[new_obj_id]['crop'] = new_crop
                obj_points[new_obj_id]['salad_features'] = salad_features
                obj_points[new_obj_id]['clip_features'] = clip_features

                # Initialize running average for new object
                if cfg.acc_features:
                    update_running_average(obj_points[new_obj_id], salad_features, clip_features)

    # At the end of main, after all processing:
    video_fps = getattr(cfg, 'video_fps', 15)
    make_video_from_visualizations(cfg.output_folder, video_filename="final_video.mp4", fps=video_fps)


if __name__ == "__main__":
    main()