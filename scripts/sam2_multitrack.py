import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt
from PIL import Image
from utils.sam2_utils import (mask_first_frame_interactive, save_sam, 
                              propagate_video, mask_first_frame)

# Configuration
SOURCE_FRAMES = './data/living_room_1/livingroom1-color_sampled'  # Path to frames directory
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

def main():
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=DEVICE)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(SOURCE_FRAMES)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_nums = [int(f.split(".")[0]) for f in frame_names]

    # take a look the first video frame
    frame_idx = 0
    frame_path = os.path.join(SOURCE_FRAMES, frame_names[frame_idx])
    ann_frame_idx = frame_idx
    
    # predictor, inference_state = mask_first_frame_interactive(predictor, video_path=SOURCE_FRAMES, frame_idx=ann_frame_idx, viz=True)
    predictor, inference_state = mask_first_frame(predictor, video_path=SOURCE_FRAMES, frame_idx=ann_frame_idx, viz=True)

    # Show the results
    # run propagation throughout the video and collect the results in a dict
    predictor, video_segments = propagate_video(predictor, inference_state, video_path=SOURCE_FRAMES)

    # render the segmentation results every few frames
    plt.close("all")
    save_sam(frame_names, frame_nums, video_segments, SOURCE_FRAMES, OUTPUT_DIR)



if __name__ == "__main__":
    main()