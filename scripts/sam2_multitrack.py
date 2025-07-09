import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
SOURCE_FRAMES = './data/living_room_1/livingroom1-color_small'  # Path to frames directory
OUTPUT_DIR = './data/sam2_res'  # Where to save results
MAX_FRAMES = 300  # Maximum frames to process
MODEL_TYPE = 'vit_b'  # SAM model type
CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'  # SAM checkpoint path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load SAM model
sam2_checkpoint = "external/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml" 

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=DEVICE)

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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def main():
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(SOURCE_FRAMES)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0
    frame_path = os.path.join(SOURCE_FRAMES, frame_names[frame_idx])
    ann_frame_idx = frame_idx
    img = Image.open(frame_path)

    # === Click collection setup ===
    clicked_points = []
    labels = []  # 1 = positive, 0 = negative (optional, defaults to 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Click to add points on frame {frame_idx}. Press 'Enter' or 'q' to finish.")
    ax.imshow(img)

    def onclick(event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            clicked_points.append((x, y))
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
    if not clicked_points:
        raise ValueError("No points were clicked!")

    all_points = np.array(clicked_points, dtype=np.float32)
    points = all_points
    ann_obj_ids = np.arange(all_points.shape[0])
    # for labels, `1` means positive click and `0` means negative click
    labels = np.ones(all_points.shape[0], np.int32)
    inference_state = predictor.init_state(video_path=SOURCE_FRAMES)
    predictor.reset_state(inference_state)
    prompts = {}
    for i, ann_obj_id in enumerate(ann_obj_ids):
        prompts[ann_obj_id] = points[i], labels[i]

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points[i, None],
            labels=labels[i, None],
        )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(SOURCE_FRAMES, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.show()

    # Show the results
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        frame_path = os.path.join(SOURCE_FRAMES, frame_names[out_frame_idx])
        frame_img = Image.open(frame_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {out_frame_idx}")
        ax.imshow(frame_img)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # --- Save mask as .npy ---
            mask_filename = f"frame{out_frame_idx:04d}_obj{out_obj_id}.npy"
            mask_path = os.path.join(OUTPUT_DIR, "masks", mask_filename)
            np.save(mask_path, out_mask)

            # --- Show mask on the frame ---
            show_mask(out_mask, ax, obj_id=out_obj_id)

        # --- Save visualization as .png ---
        vis_filename = f"frame{out_frame_idx:04d}.png"
        vis_path = os.path.join(OUTPUT_DIR, "visualizations", vis_filename)
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    main()