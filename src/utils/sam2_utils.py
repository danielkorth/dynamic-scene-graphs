import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from skimage.measure import label, regionprops
import glob

from segment import SAM2Segmenter
from scipy.ndimage import binary_erosion


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def mask_first_frame_interactive(predictor, video_path, frame_idx = 0, viz=False):
    image_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    img_path = os.path.join(video_path, image_files[frame_idx])
    img_pil = Image.open(img_path).convert("RGB")

    # === Click collection setup ===
    in_points = []
    labels = []  # 1 = positive, 0 = negative (optional, defaults to 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Click to add points on frame {frame_idx}. Press 'Enter' or 'q' to finish.")
    ax.imshow(img_pil)

    def onclick(event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            in_points.append((x, y))
            labels.append(1)  # always positive in this example
            ax.plot(x, y, 'go')  # plot green dot
            fig.canvas.draw()

    def onkey(event):
        if event.key == 'enter' or event.key == 'q':
            plt.close()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()

    # === Validate input ===
    if not in_points:
        raise ValueError("No points were clicked!")

    all_points = np.array(in_points, dtype=np.float32)
    
    points = all_points
    ann_obj_ids = np.arange(all_points.shape[0])
    # for labels, `1` means positive click and `0` means negative click
    labels = np.ones(all_points.shape[0], np.int32)
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)
    prompts = {}
    for i, ann_obj_id in enumerate(ann_obj_ids):
        prompts[ann_obj_id] = points[i], labels[i]

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=points[i, None],
            labels=labels[i, None],
        )

    # show the results on the current (interacted) frame
    if viz:
        image_files = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_path, image_files[frame_idx])))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            show_points(*prompts[out_obj_id], plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.show()

    return predictor, inference_state


def mask_first_frame(first_frame: np.ndarray, segmenter: SAM2Segmenter, viz: bool = False):
    masks = segmenter.segment(first_frame)
    print(f"Found {len(masks)} masks")

    # plt imshow the image + masks
    if viz:
        fig, ax = plt.subplots()
        ax.imshow(first_frame)
        for i, mask in enumerate(masks):
            show_mask(mask['segmentation'], ax, obj_id=i, random_color=False)
        ax.axis('off')
        ax.set_title(f"Masks overlaid on frame {0}")
        plt.tight_layout()
        plt.show()

    return masks

def sample_points_per_cc(
        comp_mask: np.ndarray,
        area_per_sample: int = 100,
        max_samples: int = 20,
        min_area: int = 20
    ) -> np.ndarray:
    """
    For each connected component in comp_mask:
      - If its area < min_area: skip it.
      - Otherwise, sample `ceil(area / area_per_sample)` points,
        but at most max_samples, at least 1.
    Returns an (N,2) array of (row, col) coords.
    """
    labels = label(comp_mask, connectivity=1)
    points = []

    for region in regionprops(labels):
        area = region.area
        if area < min_area:
            # too small to consider
            continue

        # decide how many points to sample
        num = math.ceil(area / area_per_sample)
        num = max(1, min(num, max_samples))

        coords = region.coords  # (row, col) pairs
        # randomly choose without replacement
        chosen = coords[np.random.choice(len(coords), num, replace=False)]
        points.append(chosen)

    if not points:
        return np.zeros((0, 2), dtype=int)
    return np.vstack(points)

def refine_masks_with_complement(
        img_predictor,
        img: np.ndarray,
        masks: list[dict],
        min_iou_new: float = 0.1,
        new_only : bool = False
    ):
    """
    Given an initial set of masks, compute how much of the image they cover;
    if below threshold, extract the complementary region and ask SAM2 to
    segment inside it, then merge any sufficiently new masks back in.

    Args:
        segmenter: your SAM2Segmenter (must support a mask_prompt kwarg).
        img: HxWx3 numpy array.
        masks: list of dicts with key 'segmentation' (HxW boolean mask).
        coverage_threshold: if union_coverage < this, we try to refill.
        min_iou_new: a new mask is kept only if its max IoU with existing
                     masks is < this (to avoid duplicates).

    Returns:
        masks: the updated list, including any new masks found.
        coverage: float [0-1] of original union coverage.
    """
    if new_only:
        out = []
    else:
        out = masks
    # complementary region

    if isinstance(masks, list) and isinstance(masks[0], dict):
        all_seg = np.stack([m['segmentation'] for m in masks], axis=0)  # N×H×W
    elif isinstance(masks, list) and isinstance(masks[0], np.ndarray):
        all_seg = np.stack(masks, axis=0)
    elif isinstance(masks, np.ndarray):
        all_seg = masks[None]
    else:
        raise(Exception("Bad Mask Formatting"))
    
    H, W = all_seg[0].squeeze().shape
    # 1) compute union mask
    union = np.any(all_seg, axis=0).squeeze()
    comp_mask = ~union
    # Erode the complementary mask to avoid edge bleed
    erode_radius = 20  # adjust as needed (in pixels)
    structure = np.ones((erode_radius, erode_radius), dtype=bool)
    comp_mask = binary_erosion(comp_mask, structure=structure)

    # call SAM2 on that region; assumes your segmenter supports a mask prompt
    # (if not, you can zero-out img outside comp_mask, or supply comp_mask as prompt)
    img_predictor.set_image(img)
    # bb = get_bounding_box(comp_mask)
    p_coords = sample_points_per_cc(
        comp_mask,
        area_per_sample=150,
        max_samples=15,
        min_area=80
    )
    if p_coords.shape[0] == 0:
        return out  # nothing new to try\
    
    # SAM expects shape (num_points, 1, 2) and float labels
    p_coords = p_coords[:, None, :]
    p_labels = np.ones((p_coords.shape[0], 1), dtype=np.int32)
    new_masks, scores, _ = img_predictor.predict(
        point_coords=p_coords,
        point_labels=p_labels,
        multimask_output=False,
    )
    # ------------------------------------------------------------
    # Non-Maximum Suppression on newly proposed masks
    # ------------------------------------------------------------
    # pair each mask with its score
    proposals = list(zip(new_masks, scores, p_coords.reshape(-1, 2)))
    # sort by score descending
    proposals.sort(key=lambda x: x[1], reverse=True)

    kept_new = []  # list of dict: {'segmentation': mask, 'point_coords': [...]}
    kept_segs = []  # boolean masks of kept new proposals

    def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.logical_and(a, b).sum()
        union_ = np.logical_or(a, b).sum() + 1e-8
        return intersection / union_

    for seg_mask, score, coord in proposals:
        # check IoU against already kept new masks
        if all(compute_iou(seg_mask, existing) < min_iou_new for existing in kept_segs):
            kept_segs.append(seg_mask)
            kept_new.append({
                'segmentation': seg_mask,
                'point_coords': [[int(coord[0]), int(coord[1])]]
            })

    # ------------------------------------------------------------
    # Filter out any that overlap too much with original masks
    # ------------------------------------------------------------
    final_new = []
    for nm in kept_new:
        seg = nm['segmentation']
        # compute IoU vs each existing original mask
        ious = [
            compute_iou(seg, m['segmentation'])
            for m in masks
        ]
        if max(ious, default=0) < min_iou_new:
            final_new.append(nm)
            

    # append kept new masks
    out.extend(final_new)

    # visualize_refinement(
    #     img=img,
    #     union=union,
    #     masks=masks,
    #     comp_mask=comp_mask,
    #     sampled_points=p_coords.squeeze(),
    #     out_masks=out
    # )

    return out

def visualize_refinement(img: np.ndarray,
                         union: np.ndarray,
                         masks: list[dict],
                         comp_mask: np.ndarray,
                         sampled_points: np.ndarray,
                         out_masks: list[dict]):
    """
    Visualize the steps of mask refinement:
      1) comp_mask overlaid
      2) sampled points overlaid
      3) final masks (out_masks) overlaid

    Args:
        img: HxWx3 numpy array, the original image.
        union: HxW boolean mask of original union (for context if needed).
        masks: list of dicts of original masks (each contains 'segmentation').
        comp_mask: HxW boolean array, the complementary region.
        sampled_points: numpy array of shape (N,2), point coordinates sampled in comp_mask.
        out_masks: list of dicts of final masks (each contains 'segmentation').
    """
    # helper for showing mask provided by user
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    # 1) Complementary region overlay
    ax1.imshow(img)
    show_mask(comp_mask, ax1, obj_id=0)
    ax1.set_title('Complementary Region Overlay')
    ax1.axis('off')

    # 2) Sampled points overlay
    ax2.imshow(img)
    # plot points as red dots
    y_coords = sampled_points[:, 0]
    x_coords = sampled_points[:, 1]
    ax2.scatter(x_coords, y_coords, c='red', s=50, marker='x')
    ax2.set_title('Sampled Points in Complementary Region')
    ax2.axis('off')

    # 3) Final masks overlay
    ax3.imshow(img)
    for idx, m in enumerate(out_masks):
        seg = m['segmentation']
        show_mask(seg, ax3, obj_id=idx, random_color=True)
    ax3.set_title('Final Refined Masks Overlay')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


#####################################################################################
# Functions V2
#####################################################################################

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


def detect_with_furthest(full_mask, buffer_radius, n_samples):
    safe_mask = get_dilated_mask(np.squeeze(full_mask), buffer_radius=buffer_radius)
    sampled_points = farthest_point_sampling(np.squeeze(~safe_mask), n_samples=n_samples)

    # Create the return list of dicts
    return_list = []
    for i in range(len(sampled_points)):
        return_list.append({
            'points': sampled_points[i],
            'labels': np.ones(1, dtype=np.int32)
        })
    return return_list

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
