#!/usr/bin/env python3
"""
Script to compute depth maps using MoGe-2 model for all images in a directory.
"""

import cv2
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.optimize import least_squares
from moge.model.v2 import MoGeModel

def create_alignment_mask(sensor_depth, canny_low=50, canny_high=150, dilation_kernel_size=5, dilation_iterations=2):
    """
    Create a mask for depth alignment by removing edges from the sensor depth map.
    
    Args:
        sensor_depth: Sensor depth map (numpy array)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        dilation_kernel_size: Size of dilation kernel
        dilation_iterations: Number of dilation iterations
        
    Returns:
        mask: Boolean mask where True indicates pixels to use for alignment
    """
    # Create initial mask for valid sensor depth pixels
    initial_mask = sensor_depth > 0
    
    # Detect edges in the sensor depth map
    sensor_depth_uint8 = (sensor_depth * 255).astype(np.uint8)
    edges = cv2.Canny(sensor_depth_uint8, canny_low, canny_high)
    
    # Dilate edges to remove more pixels around edges
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    
    # Create edge mask (True where edges are detected)
    edge_mask = dilated_edges > 0
    
    # Combine masks: valid pixels AND not edges
    mask = initial_mask & ~edge_mask
    
    return mask

def compute_ransac_scale_shift(pred_valid, gt_valid, max_iterations=100, threshold=0.1, min_inliers_ratio=0.5):
    """
    Compute scale and shift using RANSAC to be robust to outliers.
    
    Args:
        pred_valid: Predicted depth values (torch tensor)
        gt_valid: Ground truth depth values (torch tensor)
        max_iterations: Maximum number of RANSAC iterations
        threshold: Inlier threshold (relative error)
        min_inliers_ratio: Minimum ratio of inliers to consider a good fit
        
    Returns:
        scale: Best scale factor
        shift: Best shift value
    """
    # Ensure inputs are torch tensors
    if not isinstance(pred_valid, torch.Tensor):
        pred_valid = torch.tensor(pred_valid, dtype=torch.float32)
    if not isinstance(gt_valid, torch.Tensor):
        gt_valid = torch.tensor(gt_valid, dtype=torch.float32)
    
    # Move to CPU for numpy operations if needed
    device = pred_valid.device
    pred_cpu = pred_valid.cpu()
    gt_cpu = gt_valid.cpu()
    
    # Convert to numpy for easier manipulation
    pred_np = pred_cpu.numpy()
    gt_np = gt_cpu.numpy()
    
    n_points = len(pred_np)
    min_inliers = int(min_inliers_ratio * n_points)
    
    best_scale = 1.0
    best_shift = 0.0
    best_inliers = 0
    
    # RANSAC iterations
    for _ in range(max_iterations):
        # Randomly sample 2 points to fit a line (scale and shift)
        if n_points < 2:
            continue
            
        indices = np.random.choice(n_points, size=2, replace=False)
        pred_sample = pred_np[indices]
        gt_sample = gt_np[indices]
        
        # Fit line through these 2 points: gt = scale * pred + shift
        # Using least squares on the sample
        pred_mean = np.mean(pred_sample)
        gt_mean = np.mean(gt_sample)
        
        numerator = np.sum((pred_sample - pred_mean) * (gt_sample - gt_mean))
        denominator = np.sum((pred_sample - pred_mean) ** 2)
        
        if denominator < 1e-8:
            continue
            
        scale = numerator / denominator
        shift = gt_mean - scale * pred_mean
        
        # Apply transformation to all points
        aligned_pred = scale * pred_np + shift
        
        # Calculate residuals
        residuals = np.abs(gt_np - aligned_pred)
        relative_errors = residuals / (np.abs(gt_np) + 1e-8)  # Avoid division by zero
        
        # Count inliers (points with relative error below threshold)
        inliers = np.sum(relative_errors < threshold)
        
        # Update best model if we have more inliers
        if inliers > best_inliers and inliers >= min_inliers:
            best_inliers = inliers
            best_scale = scale
            best_shift = shift
    
    # If RANSAC failed to find a good model, fall back to least squares
    if best_inliers < min_inliers:
        # Fall back to least squares method
        pred_mean = torch.mean(pred_valid)
        gt_mean = torch.mean(gt_valid)
        
        numerator = torch.sum((pred_valid - pred_mean) * (gt_valid - gt_mean))
        denominator = torch.sum((pred_valid - pred_mean) ** 2)
        
        if denominator < 1e-8:
            best_scale = 1.0
        else:
            best_scale = numerator / denominator
        
        best_shift = gt_mean - best_scale * pred_mean
    
    return best_scale, best_shift


def compute_nonlinear_scale_shift(pred_valid, gt_valid, loss='huber', huber_scale=1.345):
    """
    Compute scale and shift using non-linear optimization with robust cost functions.
    
    Args:
        pred_valid: Predicted depth values (torch tensor)
        gt_valid: Ground truth depth values (torch tensor)
        loss: Loss function type ('huber', 'soft_l1', 'cauchy', 'arctan')
        huber_scale: Scale parameter for Huber loss (default: 1.345)
        
    Returns:
        scale: Best scale factor
        shift: Best shift value
    """
    # Ensure inputs are torch tensors and move to CPU
    if not isinstance(pred_valid, torch.Tensor):
        pred_valid = torch.tensor(pred_valid, dtype=torch.float32)
    if not isinstance(gt_valid, torch.Tensor):
        gt_valid = torch.tensor(gt_valid, dtype=torch.float32)
    
    pred_np = pred_valid.cpu().numpy()
    gt_np = gt_valid.cpu().numpy()
    
    # Define the residual function for optimization
    def residuals(params):
        scale, shift = params
        predicted = scale * pred_np + shift
        return gt_np - predicted
    
    # Initial guess using least squares
    pred_mean = np.mean(pred_np)
    gt_mean = np.mean(gt_np)
    
    numerator = np.sum((pred_np - pred_mean) * (gt_np - gt_mean))
    denominator = np.sum((pred_np - pred_mean) ** 2)
    
    if denominator < 1e-8:
        initial_scale = 1.0
    else:
        initial_scale = numerator / denominator
    
    initial_shift = gt_mean - initial_scale * pred_mean
    initial_params = [initial_scale, initial_shift]
    
    # Set bounds to prevent unreasonable values
    # Scale should be positive and reasonable
    bounds = ([0.1, -np.inf], [10.0, np.inf])
    
    try:
        # Run non-linear optimization with robust loss
        result = least_squares(
            residuals,
            initial_params,
            loss=loss,
            bounds=bounds,
            method='trf',  # Trust Region Reflective algorithm
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=1000
        )
        
        if result.success:
            scale, shift = result.x
        else:
            # Fall back to initial values if optimization fails
            scale, shift = initial_params
            
    except Exception:
        # Fall back to initial values if optimization fails
        scale, shift = initial_params
    
    return scale, shift

def align_pred_with_gt(pred_depth, gt_depth, method="nonlinear_huber"):
    """
    Align predicted depth map with ground truth depth map.
    It finds the scale and shift that minimizes the difference between the predicted and ground truth depths.
    It then applies the scale and shift to the predicted depth map.

    Args:
        pred_depth: Predicted depth map (already masked and flattened)
        gt_depth: Ground truth depth map (already masked and flattened)
        mask: Mask of valid pixels (flattened)
        
    Returns:
        aligned_pred_depth: Aligned predicted depth map
    """
    # Ensure inputs are torch tensors
    if not isinstance(pred_depth, torch.Tensor):
        pred_depth = torch.tensor(pred_depth, dtype=torch.float32)
    if not isinstance(gt_depth, torch.Tensor):
        gt_depth = torch.tensor(gt_depth, dtype=torch.float32)
    
    # Since inputs are already masked and flattened, we can use them directly
    pred_valid = pred_depth
    gt_valid = gt_depth
    
    if method == "least_squares":
        # Compute mean values
        pred_mean = torch.mean(pred_valid)
        gt_mean = torch.mean(gt_valid)

        # Compute scale factor using least squares
        numerator = torch.sum((pred_valid - pred_mean) * (gt_valid - gt_mean))
        denominator = torch.sum((pred_valid - pred_mean) ** 2)

         # Avoid division by zero
        if denominator < 1e-8:
            scale = 1.0
        else:
            scale = numerator / denominator
        
        # Compute shift
        shift = gt_mean - scale * pred_mean

    elif method == "median_ratio":
        # Compute median ratio
        median_ratio = torch.median(gt_valid / pred_valid)
        scale = median_ratio
        shift = 0.0

    elif method == "ransac":
        # Compute RANSAC scale and shift
        scale, shift = compute_ransac_scale_shift(pred_valid, gt_valid)

    elif method == "nonlinear_huber":
        # Compute non-linear optimization with Huber loss
        scale, shift = compute_nonlinear_scale_shift(pred_valid, gt_valid, loss='huber')
        
    elif method == "nonlinear_soft_l1":
        # Compute non-linear optimization with soft L1 loss
        scale, shift = compute_nonlinear_scale_shift(pred_valid, gt_valid, loss='soft_l1')
        
    elif method == "nonlinear_cauchy":
        # Compute non-linear optimization with Cauchy loss
        scale, shift = compute_nonlinear_scale_shift(pred_valid, gt_valid, loss='cauchy')
        
    elif method == "nonlinear_arctan":
        # Compute non-linear optimization with arctan loss
        scale, shift = compute_nonlinear_scale_shift(pred_valid, gt_valid, loss='arctan')

    else:
        raise ValueError(f"Invalid alignment method: {method}. Available methods: least_squares, median_ratio, ransac, nonlinear_huber, nonlinear_soft_l1, nonlinear_cauchy, nonlinear_arctan")
    
   
    
    # Apply scale and shift to the full predicted depth map
    aligned_pred_depth = scale * pred_depth + shift
    
    return aligned_pred_depth

def compute_moge_depths(input_dir, device="cuda", align_depths=False):
    """
    Compute depth maps for all images in input_dir using MoGe-2 model.
    The output directory is the same as the input directory, but with "moge_depths" appended.
    
    Args:
        input_dir: Directory containing RGB images (left*.png format)
        device: Device to run inference on ("cuda" or "cpu")
    """
    # Create output directory
    dir_name = "moge_depths" if not align_depths else "moge_aligned_depths"
    output_path = Path(input_dir).parent / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the MoGe-2 model
    print(f"Loading MoGe-2 model on {device}...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    
    # Find all RGB images
    rgb_pattern = os.path.join(input_dir, "left*.png")
    rgb_files = sorted(glob.glob(rgb_pattern))
    
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {input_dir}")
    
    print(f"Found {len(rgb_files)} images to process")
    
    # Process each image
    for rgb_file in tqdm(rgb_files, desc="Computing depths"):
        # Extract frame number from filename
        filename = os.path.basename(rgb_file)
        frame_num = int(filename.replace("left", "").replace(".png", ""))
        
        # Read and preprocess image
        input_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        if input_image is None:
            print(f"Warning: Failed to load {rgb_file}")
            continue
            
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        # Add batch dimension
        input_image = input_image.unsqueeze(0)
        
        # Infer depth
        with torch.no_grad():
            output = model.infer(input_image)
        
        # Extract depth map
        depth_map = output["depth"][0]  # Remove batch dimension

        if align_depths:
            sensor_depth = cv2.imread(os.path.join(input_dir, f"depth{frame_num:06d}.png"), cv2.IMREAD_UNCHANGED)
            sensor_depth = sensor_depth.astype(np.float32) / 1000.0
            
            # Create alignment mask (excluding edges)
            mask = create_alignment_mask(sensor_depth)
            
            # Apply mask to get valid pixels for alignment
            sensor_depth_valid = sensor_depth[mask]
            depth_map_masked = depth_map[mask]
            depth_map_masked = depth_map_masked.flatten()
            sensor_depth_valid = sensor_depth_valid.flatten()
            sensor_depth_valid = torch.tensor(sensor_depth_valid, device=device)
            
            # Align the masked depth maps (excluding edges)
            aligned_depth_masked = align_pred_with_gt(depth_map_masked, sensor_depth_valid)
            
            # Reconstruct the full depth map with aligned values
            depth_map_flat = depth_map.flatten().clone()  # Clone to make it a normal tensor
            depth_map_flat[mask.flatten()] = aligned_depth_masked
            depth_map = depth_map_flat.reshape(depth_map.shape)

        depth_map = depth_map.cpu().numpy()
        # Save depth map as PNG (16-bit)
        depth_file = output_path / f"depth{frame_num:06d}.png"
        depth_uint16 = (depth_map * 1000).astype(np.uint16)  # Convert to millimeters
        cv2.imwrite(str(depth_file), depth_uint16)
        
        print(f"Saved depth map for frame {frame_num} to {depth_file}")

def main():
    parser = argparse.ArgumentParser(description="Compute depth maps using MoGe-2")
    parser.add_argument("input_dir", help="Directory containing RGB images (left*.png)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], 
                       help="Device to run inference on")
    parser.add_argument("--align_depths", action="store_true", help="Align MoGe depths to sensor depths")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist")
    
    compute_moge_depths(args.input_dir, args.device, args.align_depths)
    print("Depth computation completed!")

if __name__ == "__main__":
    import glob
    main()
