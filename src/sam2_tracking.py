#!/usr/bin/env python3
import os
import glob
import shutil
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt

# SAM2 imports
from sam2.build_sam import build_sam2_video_predictor
from utils.sam2_utils import (mask_first_frame_interactive, save_sam, 
                              propagate_video_plain)


@hydra.main(version_base=None, config_path="../configs", config_name="sam2_tracking")
def main(cfg: DictConfig) -> None:
    """
    Combined SAM2 multitrack processing pipeline.
    
    This script combines the functionality of:
    - copy_and_run_sam2_multitrack.sh
    - select_and_convert_frames.py  
    - sam2_multitrack.py
    
    into a single unified pipeline.
    """
    
    print("SAM2 Multitrack Combined Pipeline")
    print("=" * 50)
    
    # Validate input arguments
    if not os.path.exists(cfg.source_folder):
        raise FileNotFoundError(f"Source folder not found: {cfg.source_folder}")
    
    source_images_dir = cfg.source_folder
    
    # Set device based on config
    if cfg.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.device
    
    print(f"Using device: {device}")
    print(f"Source folder: {cfg.source_folder}")
    print(f"Masks output folder: {cfg.masks_output_folder}")
    print(f"Mask images output folder: {cfg.mask_images_output_folder}")
    print(f"Stride: {cfg.stride}, Max frames: {cfg.max_frames}")
    
    # Create output directory structure
    os.makedirs(cfg.masks_output_folder, exist_ok=True)
    os.makedirs(cfg.mask_images_output_folder, exist_ok=True)
    
    # Create temporary directory for processed frames
    temp_frames_dir = cfg.temp_folder
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    try:
        # Step 1: Frame Discovery and Processing
        print("\nStep 1: Discovering and processing frames...")
        try:
            process_frames(source_images_dir, temp_frames_dir, cfg.stride, cfg.max_frames, cfg.jpeg_quality)
        except Exception as e:
            raise RuntimeError(f"Frame processing failed: {e}")
        
        # Step 2: SAM2 Model Loading
        print("\nStep 2: Loading SAM2 model...")
        try:
            # hydra.core.global_hydra.GlobalHydra.instance().clear()
            predictor = load_sam2_model(cfg.sam.sam2_checkpoint, cfg.sam.model_config, device)
        except Exception as e:
            raise RuntimeError(f"SAM2 model loading failed: {e}")
        
        # Step 3: SAM2 Processing
        print("\nStep 3: Running SAM2 segmentation...")
        try:
            run_sam2_segmentation(predictor, temp_frames_dir, cfg.masks_output_folder, cfg.mask_images_output_folder, device)
        except Exception as e:
            raise RuntimeError(f"SAM2 segmentation failed: {e}")
        
        print("\nProcessing completed successfully!")
        print(f"Masks saved to: {cfg.masks_output_folder}")
        print(f"Visualizations saved to: {cfg.mask_images_output_folder}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
        
    finally:
        # Cleanup temporary files if requested
        if cfg.cleanup_temp and os.path.exists(temp_frames_dir):
            print(f"\nCleaning up temporary files in {temp_frames_dir}")
            try:
                shutil.rmtree(temp_frames_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temporary files: {e}")


def process_frames(source_images_dir: str, output_dir: str, stride: int, max_frames: int, jpeg_quality: int) -> None:
    """
    Discover PNG files, apply filtering, and convert to sequential JPEG format.
    Equivalent to the bash script + select_and_convert_frames.py functionality.
    """
    # Discover left*.png files
    png_pattern = os.path.join(source_images_dir, "left*.png")
    png_files = sorted(glob.glob(png_pattern))
    
    if not png_files:
        raise FileNotFoundError(f"No left*.png files found in {source_images_dir}")
    
    print(f"Found {len(png_files)} PNG files")
    
    # Apply stride filtering
    selected_files = png_files[::stride]
    
    # Apply max_frames limit if specified
    if max_frames > 0:
        selected_files = selected_files[:max_frames]
    
    n_selected = len(selected_files)
    print(f"Selected {n_selected} frames (stride={stride}, max_frames={max_frames})")
    
    if n_selected == 0:
        raise ValueError("No frames selected after filtering")
    
    # Convert PNG to sequential JPEG
    print("Converting PNG files to sequential JPEG format...")
    
    for idx, png_path in enumerate(tqdm(selected_files, desc="Converting")):
        # Create sequential JPEG filename (5-digit zero-padded)
        # Use simple sequential numbering (0, 1, 2, ...) for SAM2 compatibility
        jpeg_filename = f"{idx:05d}.jpg"
        jpeg_path = os.path.join(output_dir, jpeg_filename)
        
        # Convert PNG to JPEG
        with Image.open(png_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(jpeg_path, 'JPEG', quality=jpeg_quality)
    
    print(f"Converted {n_selected} frames to JPEG format in {output_dir}")


def load_sam2_model(checkpoint_path: str, config_path: str, device: str):
    """
    Load SAM2 model with specified checkpoint and configuration.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")
    
    print(f"Loading SAM2 model from {checkpoint_path}")
    print(f"Using config: {config_path}")
    
    predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
    
    print("SAM2 model loaded successfully")
    return predictor


def run_sam2_segmentation(predictor, frames_dir: str, masks_output_dir: str, visualizations_output_dir: str, device: str) -> None:
    """
    Run SAM2 segmentation pipeline on processed frames.
    Equivalent to sam2_multitrack.py functionality.
    """
    # Scan all JPEG frame names in the directory
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    
    if not frame_names:
        raise FileNotFoundError(f"No JPEG files found in {frames_dir}")
    
    # Sort by numeric order (assumes format 00000.jpg, 00001.jpg, etc.)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_nums = [int(f.split(".")[0]) for f in frame_names]
    
    print(f"Found {len(frame_names)} JPEG frames for processing")
    
    # Start segmentation with the first frame
    frame_idx = 0
    ann_frame_idx = frame_idx
    
    print("Running interactive first frame masking...")
    predictor, inference_state = mask_first_frame_interactive(
        predictor, 
        video_path=frames_dir, 
        frame_idx=ann_frame_idx, 
        viz=True
    )
    
    print("Propagating segmentation across video...")
    predictor, video_segments = propagate_video_plain(
        predictor, 
        inference_state, 
        video_path=frames_dir
    )
    
    # Close any open plots
    plt.close("all")
    
    print("Saving segmentation results...")
    save_sam_custom(frame_names, frame_nums, video_segments, frames_dir, masks_output_dir, visualizations_output_dir)
    
    print("SAM2 segmentation completed successfully")


def save_sam_custom(frame_names, frame_nums, video_segments, video_folder, masks_output_dir, visualizations_output_dir):
    """
    Custom save function that writes masks and visualizations to separate directories.
    Based on the original save_sam function but with separate output directories.
    """
    from utils.tools import get_color_for_id
    
    for out_frame_idx in range(0, len(frame_names)):
        frame_path = os.path.join(video_folder, frame_names[out_frame_idx])
        frame_img = Image.open(frame_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"frame {frame_nums[out_frame_idx]}")
        ax.imshow(frame_img)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # --- Save mask as .npy to masks directory ---
            mask_filename = f"frame{frame_nums[out_frame_idx]:04d}_obj{out_obj_id}.npy"
            mask_path = os.path.join(masks_output_dir, mask_filename)
            np.save(mask_path, out_mask)

            # --- Show mask on the frame with consistent color ---
            color = get_color_for_id(out_obj_id)
            mask = out_mask
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * np.array([*color, 0.6]).reshape(1, 1, -1)
            ax.imshow(mask_image)

        # --- Save visualization as .png to visualizations directory ---
        vis_filename = f"frame{frame_nums[out_frame_idx]:04d}.png"
        vis_path = os.path.join(visualizations_output_dir, vis_filename)
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main() 