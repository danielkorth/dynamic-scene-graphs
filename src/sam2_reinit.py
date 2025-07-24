from utils.sam2_utils import show_mask, show_points
from sam2.build_sam import build_sam2_video_predictor
from segment import SAM2Segmenter
import hydra
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re

def create_overlapping_subsets(input_dir, output_dir, stride=5, overlap=1, subsample=1):
    """
    Create overlapping subsequences of left RGB images, with optional subsampling.
    Args:
        input_dir: Path to directory containing left*.png images
        output_dir: Path to output directory where subset folders will be created
        stride: Number of images per subset (default: 5)
        overlap: Number of overlapping images between subsets (default: 1)
        subsample: Take every Nth image (default: 1, no subsampling)
    Returns:
        List of subset directory paths
    """
    left_pattern = os.path.join(input_dir, "left*.png")
    left_files = sorted(glob.glob(left_pattern))
    left_files = left_files[::subsample]
    if not left_files:
        raise FileNotFoundError(f"No left*.png files found in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    step_size = stride - overlap
    subset_idx = 0
    start_idx = 0
    subset_paths = []
    while start_idx < len(left_files):
        end_idx = min(start_idx + stride, len(left_files))
        subset_dir = os.path.join(output_dir, f"subset_{subset_idx}")
        os.makedirs(subset_dir, exist_ok=True)
        subset_paths.append(subset_dir)
        for j, i in enumerate(range(start_idx, end_idx)):
            input_path = left_files[i]
            # Name images as 000000.jpg, 000001.jpg, ... within each subset
            jpg_filename = f"{j:06d}.jpg"
            output_path = os.path.join(subset_dir, jpg_filename)
            with Image.open(input_path) as img:
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode not in ('RGB',):
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)
        print(f"Created {subset_dir} with {end_idx - start_idx} images (indices {start_idx}-{end_idx-1})")
        subset_idx += 1
        start_idx += step_size
        if start_idx >= len(left_files):
            break
    return subset_paths

# Save results using OpenCV for speed
def save_sam_cv2(video_segments, frames_dir, masks_dir, vis_dir):
    from utils.tools import get_color_for_id
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    for out_frame_idx in video_segments.keys():
        frame_path = os.path.join(frames_dir, f"{out_frame_idx:06d}.jpg")
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            print(f"Warning: Could not load frame {frame_path}")
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        overlay_img = frame_rgb.copy().astype(np.float32)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # Save mask as .npy
            mask_filename = f"frame{out_frame_idx:06d}_obj{out_obj_id}.npy"
            mask_path = os.path.join(masks_dir, mask_filename)
            np.save(mask_path, out_mask)
            # Overlay mask
            color = get_color_for_id(out_obj_id)
            color_rgb = np.array(color) * 255
            mask = out_mask
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            elif mask.ndim == 3:
                mask = np.any(mask, axis=0)
            elif mask.ndim != 2:
                print(f"Warning: Unexpected mask shape {mask.shape} for object {out_obj_id}, skipping overlay")
                continue
            if mask.shape != overlay_img.shape[:2]:
                print(f"Warning: Mask shape {mask.shape} doesn't match image shape {overlay_img.shape[:2]} for object {out_obj_id}, skipping overlay")
                continue
            mask_bool = mask.astype(bool)
            if np.any(mask_bool):
                overlay_img[mask_bool] = overlay_img[mask_bool] * 0.4 + color_rgb * 0.6
        result_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        vis_filename = f"frame{out_frame_idx:06d}.png"
        vis_path = os.path.join(vis_dir, vis_filename)
        cv2.imwrite(vis_path, result_bgr)
    print(f"Saved {len(video_segments)} frames with masks to {masks_dir} and visualizations to {vis_dir}")

def save_points_image_cv2(image_path, points, labels, output_path):
    """
    Draw points on the image and save the result using OpenCV.
    Args:
        image_path: Path to the input image (str)
        points: numpy array of shape (N, 2) with (x, y) coordinates
        labels: numpy array of shape (N,) with integer labels (0 or 1)
        output_path: Path to save the output image (str)
    """
    import cv2
    import numpy as np
    # Load image (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    points = np.asarray(points)
    labels = np.asarray(labels)
    for idx, (pt, label) in enumerate(zip(points, labels)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for 1, Red for 0
        cv2.circle(img, (x, y), radius=6, color=color, thickness=-1)
        cv2.putText(img, str(idx), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(output_path, img)

def save_points_image_cv2_obj_id(image_path, obj_points, output_path):
    """
    Draw points on the image, coloring by obj_id, and save the result using OpenCV.
    Args:
        image_path: Path to the input image (str)
        obj_points: dict mapping obj_id to {'points': np.ndarray, 'labels': np.ndarray}
        output_path: Path to save the output image (str)
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    # Assign a unique color to each obj_id using a colormap
    obj_ids = sorted(obj_points.keys())
    cmap = plt.get_cmap('tab20')
    color_map = {obj_id: tuple(int(255*x) for x in cmap(i % 20)[:3][::-1]) for i, obj_id in enumerate(obj_ids)}
    for obj_id, data in obj_points.items():
        points = np.asarray(data['points'])
        if points.size == 0:
            continue
        color = color_map[obj_id]
        for idx, pt in enumerate(points):
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (x, y), radius=6, color=color, thickness=-1)
            cv2.putText(img, str(obj_id), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(output_path, img)

def make_video_from_visualizations(output_folder, video_filename="final_video.mp4", fps=15):

    """
    Combine all images from all visualizations_* folders into a video.
    Args:
        output_folder: Path to the output directory containing visualizations_* folders
        video_filename: Name of the output video file (mp4)
        fps: Frames per second for the video
    """
    import cv2
    import glob
    import os
    # Find all visualizations_* folders
    vis_dirs = sorted(
        [d for d in glob.glob(os.path.join(output_folder, "visualizations_*")) if os.path.isdir(d)],
        key=lambda x: int(re.findall(r"visualizations_(\d+)", x)[0])
    )
    frame_paths = []
    for vis_dir in vis_dirs:
        imgs = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
        frame_paths.extend(imgs)
    if not frame_paths:
        raise FileNotFoundError("No frames found in visualizations_* folders.")
    # Read first image to get size
    first_img = cv2.imread(frame_paths[0])
    if first_img is None:
        raise FileNotFoundError(f"Could not read image: {frame_paths[0]}")
    height, width = first_img.shape[:2]
    out_path = os.path.join(output_folder, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for img_path in frame_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        out.write(img)
    out.release()
    print(f"Video saved to {out_path} ({len(frame_paths)} frames, {fps} fps)")

def farthest_point_sampling(mask, n_samples, border=0):
    """
    Perform farthest point sampling on a 2D boolean mask, avoiding points near the border.
    Args:
        mask: 2D numpy array of bools (True = foreground)
        n_samples: number of points to sample
        border: minimum distance from the image border (in pixels, default 0)
    Returns:
        coords: (n_samples, 2) array of sampled (col, row) coordinates
    """
    # Get foreground coordinates
    coords = np.argwhere(mask)
    if border > 0:
        h, w = mask.shape
        coords = coords[
            (coords[:, 0] >= border) & (coords[:, 0] < h - border) &
            (coords[:, 1] >= border) & (coords[:, 1] < w - border)
        ]
    if len(coords) == 0:
        return np.array([])
    if n_samples > len(coords):
        return np.array([])

    # Initialize: pick a random point
    idx = np.random.choice(len(coords))
    selected = [coords[idx]]
    # Compute distances to the first point
    dists = np.linalg.norm(coords - coords[idx], axis=1)

    for _ in range(1, n_samples):
        # Select the point with the maximum distance to the set
        idx = np.argmax(dists)
        selected.append(coords[idx])
        # Update distances: for each point, keep the minimum distance to any selected point
        dists = np.minimum(dists, np.linalg.norm(coords - coords[idx], axis=1))

    selected = np.array(selected)
    # Swap columns to return (col, row) instead of (row, col)
    return selected[:, [1, 0]]

def centroid_point(mask):
    """
    Compute the centroid of a 2D boolean mask.
    Args:
        mask: 2D numpy array of bools (True = foreground)
    Returns:
        coords: (1, 2) array with (col, row) of the centroid, or empty array if mask is empty
    """
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.array([])
    centroid = coords.mean(axis=0)
    # Return as (col, row) shape
    return np.array([[centroid[1], centroid[0]]], dtype=np.float32)

# Example usage:
# safe_mask = get_safe_sampling_mask(mask, buffer_radius=5)
# Use safe_mask as your sampling region (True = safe to sample)

def get_safe_sampling_mask(mask, buffer_radius):
    """
    Dilate the mask by buffer_radius pixels and return a boolean mask where True indicates safe-to-sample points (i.e., at least buffer_radius away from the masked region).
    Args:
        mask: 2D numpy array of bools (True = masked/foreground)
        buffer_radius: int, minimum distance from masked region (in pixels)
    Returns:
        safe_mask: 2D numpy array of bools (True = safe to sample)
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    if buffer_radius <= 0:
        return ~mask  # All non-masked points are safe

    # Create a circular structuring element
    y, x = np.ogrid[-buffer_radius:buffer_radius+1, -buffer_radius:buffer_radius+1]
    selem = (x**2 + y**2) <= buffer_radius**2
    # Dilate the mask
    dilated = binary_dilation(mask, structure=selem)
    # Safe region: not in dilated mask
    safe_mask = ~dilated
    return safe_mask

# Example usage:
# dilated_mask = get_dilated_mask(mask, buffer_radius=5)
# Use dilated_mask as your expanded mask (True = masked/foreground, expanded)

def get_dilated_mask(mask, buffer_radius):
    """
    Dilate the mask by buffer_radius pixels and return the dilated mask.
    Args:
        mask: 2D numpy array of bools (True = masked/foreground)
        buffer_radius: int, dilation radius in pixels
    Returns:
        dilated_mask: 2D numpy array of bools (True = masked/foreground, expanded)
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    if buffer_radius <= 0:
        return mask

    y, x = np.ogrid[-buffer_radius:buffer_radius+1, -buffer_radius:buffer_radius+1]
    selem = (x**2 + y**2) <= buffer_radius**2
    dilated_mask = binary_dilation(mask, structure=selem)
    return dilated_mask

@hydra.main(config_path="../configs", config_name="sam2_reinit.yaml")
def main(cfg):

    # load and subsample images
    subsets = create_overlapping_subsets(cfg.source_folder, cfg.output_folder, cfg.stride, cfg.overlap, cfg.subsample)

    # load images
    predictor = build_sam2_video_predictor(
        config_file = cfg.sam.model_config,
        ckpt_path = cfg.sam.sam2_checkpoint,
        device = "cuda"
    )

    segmenter = SAM2Segmenter(sam2_checkpoint = cfg.sam.sam2_checkpoint, model_cfg = cfg.sam.model_config)

    # load first image
    first_image_path = subsets[0] + "/000000.jpg"
    img = Image.open(first_image_path)
    img_np = np.array(img)
    
    masks = segmenter.segment(img_np)
    print(f"Segmented first image: {first_image_path}")
    print(f"Found {len(masks)} masks")

    # plt imshow the image + masks
    if cfg.viz:
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        for i, mask in enumerate(masks):
            show_mask(mask['segmentation'], ax, obj_id=i, random_color=False)
        ax.axis('off')
        ax.set_title(f"Masks overlaid on frame {0}")
        plt.tight_layout()
        plt.show()

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

        for obj_id, points in obj_points.items():
            # if we have mask, transfer it. otherwise, use points.
            if points['mask'] is not None:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=points['mask']
                )
                continue
            # skip empty
            elif len(points['points']) > 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points['points'],
                    labels=points['labels'],
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
            # sampled_points = farthest_point_sampling(np.squeeze(mask), 1)
            # sampled_points = centroid_point(np.squeeze(mask))
            # obj_points[obj_id]['points'] = sampled_points.astype(np.float32)
            # obj_points[obj_id]['labels'] = np.ones(len(sampled_points), dtype=np.int32)
            obj_points[obj_id]['mask'] = mask.squeeze()
            full_mask += mask

        # save_points_image_cv2_obj_id(os.path.join(subsets[i], f"{len(video_segments)-1:06d}.jpg"), obj_points, os.path.join(cfg.output_folder, f"frame_{i}_obj_id.png"))

        # save the mask image
        cv2.imwrite(os.path.join(cfg.output_folder, f"frame_{i}_full_mask.png"), np.squeeze(full_mask) * 255)
        safe_mask = get_dilated_mask(np.squeeze(full_mask), buffer_radius=cfg.mask_buffer_radius)
        # save the safe mask image
        safe_mask = ~safe_mask
        cv2.imwrite(os.path.join(cfg.output_folder, f"frame_{i}_safe_mask.png"), np.squeeze(safe_mask) * 255)
        sampled_points = farthest_point_sampling(np.squeeze(safe_mask), cfg.point_sampling_points)
        # add new categories
        next_obj_id = len(obj_points)
        for obj_id_new, point in enumerate(sampled_points):
            new_obj_id = next_obj_id + obj_id_new
            obj_points[new_obj_id]['points'] = point.astype(np.float32).reshape(-1, 2)
            obj_points[new_obj_id]['labels'] = np.ones(1, dtype=np.int32)

        save_points_image_cv2_obj_id(os.path.join(subsets[i], f"{len(video_segments)-1:06d}.jpg"), obj_points, os.path.join(cfg.output_folder, f"frame_{i}_obj_id_new.png"))

    # At the end of main, after all processing:
    video_fps = getattr(cfg, 'video_fps', 15)
    make_video_from_visualizations(cfg.output_folder, video_filename="final_video.mp4", fps=video_fps)


if __name__ == "__main__":
    main()