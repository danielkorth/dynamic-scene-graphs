import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from skimage.measure import label, regionprops
import random

from sam2.build_sam import build_sam2_video_predictor, build_sam2_camera_predictor
from segment import SAM2Segmenter
from utils.tools import get_bounding_box, sample_points_in_mask, mask_union_and_coverage
from tqdm import tqdm
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
    ax.imshow(img)

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

def mask_first_frame_multi(predictor,
                    video_path: str,
                    frame_idx: int = 0,
                    viz = False
                    ) :
    """
    Automatically generate masks on the first frame and initialize tracking.

    Args:
        predictor: SAM2 predictor with auto-mask capability.
        video_path: Path to folder of frames named "%05d.jpg".
        frame_idx: Index of the frame to mask.
        iou_threshold: Minimum IoU to merge overlapping masks.
        point_sampling_method: How to sample a point from each mask ("centroid" or "random").
    Returns:
        predictor: Reset predictor ready for tracking.
        inference_state: Initialized inference state with object prompts added.
    """
    # Load image
    image_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    img_path = os.path.join(video_path, image_files[frame_idx])
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    coverage_threshold = 0.0

    # 1) Automatic mask generation
    # predictor.generate_auto_masks should return masks as [N, H, W] boolean or logits
    segmenter = SAM2Segmenter(
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt",
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
    )
    masks = segmenter.segment(img_np)

    union, coverage = mask_union_and_coverage(masks)
    if coverage < coverage_threshold:
        masks = refine_masks_with_complement(segmenter.mask_generator.predictor, img_np, union, masks)
    print(f"Image coverage was {coverage:.1%}; now have {len(masks)} masks.")

    if viz:
        # 1.5) Visualization
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        for i, mask in enumerate(masks):
            show_mask(mask['segmentation'], ax, obj_id=i, random_color=True)
        ax.axis('off')
        ax.set_title(f"Masks overlaid on frame {frame_idx}")
        plt.tight_layout()
        plt.show()

    # 2) Initialize inference state
    predictor.load_first_frame(img_np)

    # 3) For each mask, compute a representative point and feed as prompt
    for obj_id, mask in enumerate(masks):

        points = mask['point_coords']
        labels = np.ones(len(points), dtype=np.int32)

        # Add to predictor state
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            cond_state_idx=len(predictor.condition_states)-1,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

    return predictor, None

def mask_first_frame(predictor,
                    video_path: str,
                    frame_idx: int = 0,
                    viz = False
                    ) :
    """
    Automatically generate masks on the first frame and initialize tracking.

    Args:
        predictor: SAM2 predictor with auto-mask capability.
        video_path: Path to folder of frames named "%05d.jpg".
        frame_idx: Index of the frame to mask.
        iou_threshold: Minimum IoU to merge overlapping masks.
        point_sampling_method: How to sample a point from each mask ("centroid" or "random").
    Returns:
        predictor: Reset predictor ready for tracking.
        inference_state: Initialized inference state with object prompts added.
    """
    # Load image
    image_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    img_path = os.path.join(video_path, image_files[frame_idx])
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    coverage_threshold = 0.0

    # 1) Automatic mask generation
    # predictor.generate_auto_masks should return masks as [N, H, W] boolean or logits
    segmenter = SAM2Segmenter(
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt",
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
    )
    masks = segmenter.segment(img_np)

    union, coverage = mask_union_and_coverage(masks)
    if coverage < coverage_threshold:
        masks = refine_masks_with_complement(segmenter.mask_generator.predictor, img_np, union, masks)
    print(f"Image coverage was {coverage:.1%}; now have {len(masks)} masks.")

    if viz:
        # 1.5) Visualization
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        for i, mask in enumerate(masks):
            show_mask(mask['segmentation'], ax, obj_id=i, random_color=True)
        ax.axis('off')
        ax.set_title(f"Masks overlaid on frame {frame_idx}")
        plt.tight_layout()
        plt.show()

    # 2) Initialize inference state
    predictor.load_first_frame(img_np)

    # 3) For each mask, compute a representative point and feed as prompt
    for obj_id, mask in enumerate(masks):

        points = mask['point_coords']
        labels = np.ones(len(points), dtype=np.int32)

        # Add to predictor state
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

    return predictor, None


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
        pass
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

# def propagate_video(predictor, inference_state, video_path):
#     video_segments = {}  # video_segments contains the per-frame segmentation results
#     coverage_threshold = 0.6
#     single_frame_seg = SAM2Segmenter(
#         sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt",
#         model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml" 
#     )
#     img_names = os.listdir(video_path)
#     img_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#     for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#         frame_masks = []
#         video_segments[out_frame_idx] = {}
#         last_idx = out_obj_ids[-1]

#         for i, out_obj_id in enumerate(out_obj_ids):
#             mask = (out_mask_logits[i] > 0.0).cpu().numpy()
#             frame_masks.append({"segmentation": mask})
#             video_segments[out_frame_idx].update({out_obj_id: mask})

#         # NOT WORKING YET
#         union, coverage = mask_union_and_coverage(frame_masks)
#         if coverage < coverage_threshold:
#             img_pil = Image.open(os.path.join(video_path, f"{img_names[out_frame_idx]:05}")).convert("RGB")
#             img_np = np.array(img_pil)
#             masks = refine_masks_with_complement(single_frame_seg.mask_generator.predictor, img_np, union, frame_masks, new_only=True)

#             # Add to predictor state
#             for new_mask in masks:
#                 last_idx = last_idx+1
#                 _, obj_ids, mask_logits = predictor.add_new_mask(
#                     inference_state=inference_state,
#                     frame_idx=out_frame_idx,
#                     obj_id=last_idx,
#                     mask = new_mask["segmentation"].squeeze(),
#                 )
#                 print("hi")
            
#     return predictor, video_segments

def propagate_video(predictor, inference_state, video_path, viz = False):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    predictors = [predictor]
    coverage_threshold = 0.6
    single_frame_seg = SAM2Segmenter(
        sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt",
        model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml" 
    )
    img_names = os.listdir(video_path)
    img_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    for frame_idx in tqdm(range(len(img_names)), desc="propagate in video"):
        frame_masks = []
        video_segments[frame_idx] = {}

        img_pil = Image.open(os.path.join(video_path, f"{img_names[frame_idx]:05}")).convert("RGB")
        img_np = np.array(img_pil)

        last_predictor_id = 0
        for pred_idx, pred in enumerate(predictors):
            out_obj_ids, out_mask_logits = pred.track(img_np)
            last_idx = out_obj_ids[-1]

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                frame_masks.append({"segmentation": mask})
                video_segments[frame_idx].update({last_predictor_id + out_obj_id: mask})

            last_predictor_id = last_idx
        
        if viz:
            # 1.5) Visualization
            fig, ax = plt.subplots()
            ax.imshow(img_np)
            for i, mask in enumerate(frame_masks):
                show_mask(mask['segmentation'], ax, obj_id=i, random_color=False)
            ax.axis('off')
            ax.set_title(f"Masks overlaid on frame {frame_idx}")
            plt.tight_layout()
            plt.show()

        sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"
        new_p = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=predictors[0].device)
        new_p.load_first_frame(img_np)

        masks = refine_masks_with_complement(single_frame_seg.mask_generator.predictor, img_np, frame_masks, new_only=True)
        # Add to predictor state
        for new_idx, new_mask in enumerate(masks):
            points = new_mask['point_coords']
            labels = np.ones(len(points), dtype=np.int32)

            # Add to predictor state
            _, out_obj_ids, out_mask_logits = new_p.add_new_points(
                frame_idx=0,
                obj_id=new_idx,
                points=points,
                labels=labels,
            ) 

            predictors.append(new_p)     
            
    return predictor, video_segments

def propagate_video_multi(predictor, inference_state, video_path, viz = False):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    predictors = [predictor]
    coverage_threshold = 0.6
    single_frame_seg = SAM2Segmenter(
        sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt",
        model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml" 
    )
    img_names = os.listdir(video_path)
    img_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    for frame_idx in tqdm(range(len(img_names)), desc="propagate in video"):
        frame_masks = []
        video_segments[frame_idx] = {}

        img_pil = Image.open(os.path.join(video_path, f"{img_names[frame_idx]:05}")).convert("RGB")
        img_np = np.array(img_pil)

        last_predictor_id = 0
        for pred_idx, pred in enumerate(predictors):
            out_obj_ids, out_mask_logits = pred.track(img_np)
            last_idx = out_obj_ids[-1]

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                frame_masks.append({"segmentation": mask})
                video_segments[frame_idx].update({last_predictor_id + out_obj_id: mask})

            last_predictor_id = last_idx
        
        if viz:
            # 1.5) Visualization
            fig, ax = plt.subplots()
            ax.imshow(img_np)
            for i, mask in enumerate(frame_masks):
                show_mask(mask['segmentation'], ax, obj_id=i, random_color=False)
            ax.axis('off')
            ax.set_title(f"Masks overlaid on frame {frame_idx}")
            plt.tight_layout()
            plt.show()

        # sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
        # model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"
        # new_p = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=predictors[0].device)
        # new_p.load_first_frame(img_np)

        pred.load_first_frame(img_np)
        masks = refine_masks_with_complement(single_frame_seg.mask_generator.predictor, img_np, frame_masks, new_only=True)
        # Add to predictor state
        for new_idx, new_mask in enumerate(masks):
            points = new_mask['point_coords']
            labels = np.ones(len(points), dtype=np.int32)

            # Add to predictor state
            _, out_obj_ids, out_mask_logits = pred.add_new_points(
                cond_state_idx=frame_idx+1,
                frame_idx=0,
                obj_id= random.randint(0, 1000),
                points=points,
                labels=labels,
            ) 

            # predictors.append(new_p)     
            
    return predictor, video_segments

def save_sam(frame_names, frame_nums, video_segments, video_folder, output_dir):
    for out_frame_idx in range(0, len(frame_names)):
        frame_path = os.path.join(video_folder, frame_names[out_frame_idx])
        frame_img = Image.open(frame_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {out_frame_idx}")
        ax.imshow(frame_img)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # --- Save mask as .npy ---
            mask_filename = f"frame{frame_nums[out_frame_idx]:04d}_obj{out_obj_id}.npy"
            mask_path = os.path.join(output_dir, "masks", mask_filename)
            np.save(mask_path, out_mask)

            # --- Show mask on the frame ---
            show_mask(out_mask, ax, obj_id=out_obj_id)

        # --- Save visualization as .png ---
        vis_filename = f"frame{frame_nums[out_frame_idx]:04d}.png"
        vis_path = os.path.join(output_dir, "visualizations", vis_filename)
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()


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
