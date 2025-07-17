#!/usr/bin/env python3
"""
Script to undistort ZED camera RGB and depth images.

Usage:
    python src/undistort.py <input_directory>
    python src/undistort.py <input_directory> --crop

Examples:
    python src/undistort.py data/zed/new/images
    python src/undistort.py data/zed/new/images --crop
    
The --crop option removes black regions created by undistortion, producing 
clean rectangular images with only valid pixel data.
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_loading import load_camera_intrinsics, get_camera_matrix, get_distortion_coeffs


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Undistort ZED camera RGB and depth images')
    parser.add_argument('input_dir', help='Input directory containing left*.png and depth*.png images')
    parser.add_argument('--config', default='data/SN35693142.conf', 
                       help='Camera configuration file (default: data/SN35693142.conf)')
    parser.add_argument('--output-suffix', default='_undistorted',
                       help='Suffix to add to output directory name (default: _undistorted)')
    parser.add_argument('--camera', default='left', choices=['left', 'right'],
                       help='Camera to use (default: left)')
    parser.add_argument('--resolution', default='2K', choices=['2K', 'FHD', 'HD', 'VGA'],
                       help='Camera resolution (default: 2K)')
    parser.add_argument('--crop', action='store_true',
                       help='Crop undistorted images to remove black regions (creates rectangular images with only valid pixels)')
    
    return parser.parse_args()


def find_image_files(input_dir):
    """Find all RGB and depth image files in the input directory."""
    rgb_pattern = os.path.join(input_dir, "left*.png")
    depth_pattern = os.path.join(input_dir, "depth*.png")
    
    rgb_files = sorted(glob.glob(rgb_pattern))
    depth_files = sorted(glob.glob(depth_pattern))
    
    print(f"Found {len(rgb_files)} RGB images and {len(depth_files)} depth images")
    
    return rgb_files, depth_files


def undistort_image(image, camera_matrix, dist_coeffs, output_size=None):
    """Undistort a single image using camera parameters."""
    if output_size is None:
        h, w = image.shape[:2]
        output_size = (w, h)
    
    # Generate optimal camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, output_size, 1, output_size
    )
    
    # Undistort the image
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return undistorted, new_camera_matrix, roi


def crop_to_roi(image, roi):
    """Crop image to ROI to remove black regions from undistortion.
    
    Args:
        image: Undistorted image
        roi: Region of Interest tuple (x, y, w, h) from getOptimalNewCameraMatrix
        
    Returns:
        cropped_image: Image cropped to valid region only
    """
    x, y, w, h = roi
    
    # Handle case where ROI is invalid (all zeros)
    if w <= 0 or h <= 0:
        print("Warning: Invalid ROI detected, returning original image")
        return image
    
    # Crop the image to the ROI
    cropped = image[y:y+h, x:x+w]
    return cropped


def process_images(rgb_files, depth_files, camera_matrix, dist_coeffs, output_dir, crop=False):
    """Process and undistort all RGB and depth images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to track ROI info for reporting
    roi_info = None
    
    # Process RGB images
    print("Processing RGB images...")
    for rgb_file in tqdm(rgb_files, desc="RGB images"):
        try:
            # Load RGB image
            rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            if rgb_image is None:
                print(f"Warning: Failed to load {rgb_file}")
                continue
            
            # Undistort RGB image
            undistorted_rgb, _, roi = undistort_image(rgb_image, camera_matrix, dist_coeffs)
            
            # Store ROI info for reporting (use first valid ROI)
            if roi_info is None and roi[2] > 0 and roi[3] > 0:
                roi_info = {
                    'original_size': f"{rgb_image.shape[1]}x{rgb_image.shape[0]}",
                    'roi': roi,
                    'cropped_size': f"{roi[2]}x{roi[3]}"
                }
            
            # Apply cropping if requested
            if crop:
                undistorted_rgb = crop_to_roi(undistorted_rgb, roi)
            
            # Save undistorted (and possibly cropped) RGB image
            filename = os.path.basename(rgb_file)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, undistorted_rgb)
            
        except Exception as e:
            print(f"Error processing {rgb_file}: {e}")
            continue
    
    # Process depth images
    print("Processing depth images...")
    for depth_file in tqdm(depth_files, desc="Depth images"):
        try:
            # Load depth image
            depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Warning: Failed to load {depth_file}")
                continue
            
            # Undistort depth image
            undistorted_depth, _, roi = undistort_image(depth_image, camera_matrix, dist_coeffs)
            
            # Apply cropping if requested
            if crop:
                undistorted_depth = crop_to_roi(undistorted_depth, roi)
            
            # Save undistorted (and possibly cropped) depth image
            filename = os.path.basename(depth_file)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, undistorted_depth)
            
        except Exception as e:
            print(f"Error processing {depth_file}: {e}")
            continue
    
    # Print ROI information if cropping was applied
    if crop and roi_info:
        print(f"\nCropping applied:")
        print(f"  Original image size: {roi_info['original_size']}")
        print(f"  ROI (x,y,w,h): {roi_info['roi']}")
        print(f"  Cropped image size: {roi_info['cropped_size']}")
        print(f"  Removed black regions from undistortion")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        sys.exit(1)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Config file: {args.config}")
    
    # Load camera intrinsics
    try:
        intrinsics = load_camera_intrinsics(args.config, args.camera, args.resolution)
        camera_matrix = get_camera_matrix(intrinsics)
        dist_coeffs = get_distortion_coeffs(intrinsics)
        
        print(f"Loaded camera intrinsics for {args.camera.upper()}_CAM_{args.resolution.upper()}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coeffs: {dist_coeffs}")
        
    except Exception as e:
        print(f"Error loading camera intrinsics: {e}")
        sys.exit(1)
    
    # Find image files
    rgb_files, depth_files = find_image_files(args.input_dir)
    
    if not rgb_files and not depth_files:
        print("No image files found in input directory")
        sys.exit(1)
    
    # Create output directory
    if args.crop:
        output_suffix = args.output_suffix.replace('_undistorted', '_undistorted_crop')
    else:
        output_suffix = args.output_suffix
    
    output_dir = args.input_dir + output_suffix
    print(f"Output directory: {output_dir}")
    
    # Process images
    process_images(rgb_files, depth_files, camera_matrix, dist_coeffs, output_dir, args.crop)
    
    if args.crop:
        print(f"Undistortion and cropping complete! Processed images saved to: {output_dir}")
    else:
        print(f"Undistortion complete! Processed images saved to: {output_dir}")


if __name__ == "__main__":
    main()

