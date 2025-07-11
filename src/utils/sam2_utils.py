import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

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


def mask_first_frame(predictor, in_points, frame_idx, video_path, viz=False):
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
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_path, f"{frame_idx:05}.jpg")))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            show_points(*prompts[out_obj_id], plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.show()

    return predictor, inference_state

def propagate_video(predictor, inference_state):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
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