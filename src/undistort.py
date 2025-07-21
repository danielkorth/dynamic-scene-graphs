#!/usr/bin/env python3
"""
Script to undistort ZED camera RGB and depth images.

Usage:
    python src/undistort.py recording=<recording_name>
    python src/undistort.py recording=<recording_name> undistort=true

Examples:
    python src/undistort.py recording=new
    python src/undistort.py recording=new undistort=true
    
When undistort=true, cropping is automatically enabled to remove black regions 
created by undistortion, producing clean rectangular images with only valid pixel data.
"""

import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# Add src to path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_loading import load_camera_intrinsics, get_camera_matrix, get_distortion_coeffs


def find_image_files(input_dir):
    """Find all RGB and depth image files in the input directory."""
    rgb_pattern = os.path.join(input_dir, "left*.png")
    depth_pattern = os.path.join(input_dir, "depth*.png")
    
    rgb_files = sorted(glob.glob(rgb_pattern))
    depth_files = sorted(glob.glob(depth_pattern))
    
    print(f"Found {len(rgb_files)} RGB images and {len(depth_files)} depth images")
    
    return rgb_files, depth_files


def undistort_rgb_image(image, camera_matrix, dist_coeffs, output_size=None):
    """Undistort an RGB image using standard OpenCV undistortion."""
    if output_size is None:
        h, w = image.shape[:2]
        output_size = (w, h)
    
    # Generate optimal camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, output_size, 1, output_size
    )
    
    # Undistort the image using standard method (good for RGB)
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return undistorted, new_camera_matrix, roi


def undistort_depth_image(image, camera_matrix, dist_coeffs, output_size=None):
    """Undistort a depth image using nearest-neighbor interpolation to preserve depth values."""
    if output_size is None:
        h, w = image.shape[:2]
        output_size = (w, h)
    
    # Generate optimal camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, output_size, 1, output_size
    )
    
    # Generate undistortion and rectification maps
    # Use identity matrix for rotation (no rectification, just undistortion)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, output_size, cv2.CV_32FC1
    )
    
    # Undistort the depth image using nearest-neighbor interpolation
    # This preserves original depth values and avoids smooth edges
    undistorted = cv2.remap(
        image, map1, map2, interpolation=cv2.INTER_NEAREST
    )
    
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


def update_intrinsics_for_crop(camera_matrix, roi):
    """Update camera intrinsics after cropping to account for principal point shift.
    
    Args:
        camera_matrix: 3x3 camera matrix from undistortion
        roi: Region of Interest tuple (x, y, w, h) used for cropping
        
    Returns:
        updated_camera_matrix: Camera matrix with adjusted principal point
    """
    x, y, w, h = roi
    
    # Handle invalid ROI
    if w <= 0 or h <= 0:
        return camera_matrix.copy()
    
    # Copy the camera matrix
    updated_matrix = camera_matrix.copy()
    
    # Update principal point by subtracting crop offset
    updated_matrix[0, 2] -= x  # cx -= x_offset
    updated_matrix[1, 2] -= y  # cy -= y_offset
    
    return updated_matrix


def save_intrinsics(camera_matrix, output_dir):
    """Save camera intrinsics to intrinsics.txt file.
    
    Args:
        camera_matrix: 3x3 camera matrix
        output_dir: Directory to save intrinsics file
    
    Returns:
        intrinsics_file: Path to saved intrinsics file
    """
    intrinsics_file = os.path.join(output_dir, "intrinsics.txt")
    with open(intrinsics_file, 'w') as f:
        f.write(f"{camera_matrix[0, 0]:.6f}\n")  # fx
        f.write(f"{camera_matrix[1, 1]:.6f}\n")  # fy  
        f.write(f"{camera_matrix[0, 2]:.6f}\n")  # cx
        f.write(f"{camera_matrix[1, 2]:.6f}\n")  # cy
    
    return intrinsics_file


def get_final_camera_matrix(rgb_files, depth_files, camera_matrix, dist_coeffs, crop):
    """Get the final camera matrix after undistortion and optional cropping.
    
    Args:
        rgb_files: List of RGB image files
        depth_files: List of depth image files  
        camera_matrix: Original camera matrix
        dist_coeffs: Distortion coefficients
        crop: Whether cropping will be applied
        
    Returns:
        final_camera_matrix: Camera matrix after undistortion and optional cropping
    """
    # Use first available image to determine final camera matrix
    test_image = None
    undistort_func = None
    
    if rgb_files:
        test_image = cv2.imread(rgb_files[0], cv2.IMREAD_COLOR)
        undistort_func = undistort_rgb_image
    elif depth_files:
        test_image = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
        undistort_func = undistort_depth_image
    
    if test_image is None:
        return camera_matrix
    
    # Get undistorted camera matrix and ROI
    _, new_camera_matrix, roi = undistort_func(test_image, camera_matrix, dist_coeffs)
    
    # Apply cropping adjustment if needed
    if crop:
        return update_intrinsics_for_crop(new_camera_matrix, roi)
    else:
        return new_camera_matrix



def process_images(rgb_files, depth_files, camera_matrix, dist_coeffs, output_dir, crop=False):
    """Process and undistort all RGB and depth images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to track ROI info for reporting
    roi_info = None
    updated_camera_matrix = None
    
    # Process RGB images
    print("Processing RGB images using standard undistortion...")
    for rgb_file in tqdm(rgb_files, desc="RGB images"):
        try:
            # Load RGB image
            rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            if rgb_image is None:
                print(f"Warning: Failed to load {rgb_file}")
                continue
            
            # Undistort RGB image using standard method
            undistorted_rgb, new_camera_matrix, roi = undistort_rgb_image(rgb_image, camera_matrix, dist_coeffs)
            
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
                # Update camera matrix for cropping (only need to do this once)
                if updated_camera_matrix is None:
                    updated_camera_matrix = update_intrinsics_for_crop(new_camera_matrix, roi)
            
            # Save undistorted (and possibly cropped) RGB image
            filename = os.path.basename(rgb_file)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, undistorted_rgb)
            
        except Exception as e:
            print(f"Error processing RGB image {rgb_file}: {e}")
            continue
    
    # Process depth images
    print("Processing depth images using nearest-neighbor interpolation...")
    for depth_file in tqdm(depth_files, desc="Depth images"):
        try:
            # Load depth image
            depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Warning: Failed to load {depth_file}")
                continue
            
            # Undistort depth image using nearest-neighbor method
            undistorted_depth, new_camera_matrix_depth, roi = undistort_depth_image(depth_image, camera_matrix, dist_coeffs)
            
            # Apply cropping if requested
            if crop:
                undistorted_depth = crop_to_roi(undistorted_depth, roi)
                # Update camera matrix for cropping (only need to do this once)  
                if updated_camera_matrix is None:
                    updated_camera_matrix = update_intrinsics_for_crop(new_camera_matrix_depth, roi)
            
            # Save undistorted (and possibly cropped) depth image
            filename = os.path.basename(depth_file)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, undistorted_depth)
            
        except Exception as e:
            print(f"Error processing depth image {depth_file}: {e}")
            continue
    
    # Print ROI information if cropping was applied
    if crop and roi_info:
        print(f"\nCropping applied:")
        print(f"  Original image size: {roi_info['original_size']}")
        print(f"  ROI (x,y,w,h): {roi_info['roi']}")
        print(f"  Cropped image size: {roi_info['cropped_size']}")
        print(f"  Removed black regions from undistortion")
    
    # Note: Intrinsics will be saved at the end of main() function
    
    # Print method summary
    print(f"\nUndistortion methods used:")
    print(f"  RGB images: Standard cv2.undistort() with bilinear interpolation")
    print(f"  Depth images: cv2.remap() with nearest-neighbor interpolation (preserves depth values)")


@hydra.main(config_path="../configs", config_name="undistort")
def main(cfg: DictConfig):
    """Main function."""
    # Handle missing config parameters with defaults
    recording = cfg.get('recording', None)
    if recording is None:
        print("Error: recording must be provided. Use: python src/undistort.py recording=<recording_name>")
        sys.exit(1)

    # crop is automatically true when undistort is true
    crop = cfg.undistort
    output_suffix = "_undistorted"
    
    # Validate input directory
    if not os.path.exists(cfg.input_dir):
        print(f"Error: Input directory '{cfg.input_dir}' does not exist")
        sys.exit(1)
    
    # Validate config file
    if not os.path.exists(cfg.config_file):
        print(f"Error: Config file '{cfg.config_file}' does not exist")
        sys.exit(1)
    
    print(f"Input directory: {cfg.input_dir}")
    print(f"Config file: {cfg.config_file}")
    
    # Load camera intrinsics
    try:
        intrinsics = load_camera_intrinsics(cfg.config_file, cfg.camera, cfg.resolution)
        camera_matrix = get_camera_matrix(intrinsics)
        dist_coeffs = get_distortion_coeffs(intrinsics)
        
        print(f"Loaded camera intrinsics for {cfg.camera.upper()}_CAM_{cfg.resolution.upper()}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coeffs: {dist_coeffs}")
        
    except Exception as e:
        print(f"Error loading camera intrinsics: {e}")
        sys.exit(1)
    
    # Find image files
    rgb_files, depth_files = find_image_files(cfg.input_dir)
    
    if not rgb_files and not depth_files:
        print("No image files found in input directory")
        sys.exit(1)
    
    # Create output directory
    if crop:
        output_suffix = output_suffix.replace('_undistorted', '_undistorted_crop')
    
    in_dir = cfg.input_dir if not cfg.input_dir.endswith('/') else cfg.input_dir[:-1]
    output_dir = in_dir + output_suffix
    print(f"Output directory: {output_dir}")
    
    # Process images
    process_images(rgb_files, depth_files, camera_matrix, dist_coeffs, output_dir, crop)
    
    # Save final camera intrinsics
    final_camera_matrix = get_final_camera_matrix(rgb_files, depth_files, camera_matrix, dist_coeffs, crop)
    intrinsics_file = save_intrinsics(final_camera_matrix, output_dir)
    
    status = "cropped" if crop else "undistorted"
    print(f"\n✓ Camera intrinsics ({status}) saved to: {intrinsics_file}")
    
    if crop:
        print(f"Undistortion and cropping complete! Processed images saved to: {output_dir}")
    else:
        print(f"Undistortion complete! Processed images saved to: {output_dir}")


if __name__ == "__main__":
    main()

