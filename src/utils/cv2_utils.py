import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def unproject_image(depth_image, K, rvec, tvec, dist=None, mask=None):
    """
    Unproject an entire depth image to 3D world coordinates with optional distortion correction.
    
    Args:
        depth_image: Depth image (H, W) in mm
        K: Camera intrinsic matrix (3x3)
        rvec: Rotation vector (camera to world)
        tvec: Translation vector (camera to world)
        dist: Optional distortion coefficients (if None, uses pinhole projection)
        mask: Optional binary mask (H, W) to select pixels to unproject
    
    Returns:
        points_3d: 3D points in world coordinates (N, 3) where N is number of valid pixels
        pixel_coords: Corresponding pixel coordinates (N, 2) for the 3D points
    """
    h, w = depth_image.shape
    
    # Create coordinate grids
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply mask if provided
    if mask is not None:
        valid_mask = (mask > 0) & (depth_image > 0)
    else:
        valid_mask = depth_image > 0
    
    # Get valid pixel coordinates and depths
    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]
    depth_valid = depth_image[valid_mask]
    
    # Stack into (N, 2) array
    pixel_coords = np.column_stack([u_valid, v_valid]).astype(np.float32)
    
    if dist is not None:
        # Apply distortion correction
        undistorted = cv2.undistortPoints(pixel_coords, K, dist)
        undistorted = undistorted.reshape(-1, 2)
    else:
        # Regular pinhole projection - convert pixel coordinates to normalized coordinates
        # Apply inverse camera matrix: normalized = K^-1 * [u, v, 1]
        K_inv = np.linalg.inv(K)
        pixel_coords_homo = np.column_stack([pixel_coords, np.ones(len(pixel_coords))])
        undistorted = (K_inv @ pixel_coords_homo.T).T[:, :2]
    
    # Make homogeneous (add z=1)
    undistorted_homo = np.concatenate([undistorted, np.ones((undistorted.shape[0], 1))], axis=1)
    
    # Scale by depth (convert mm to meters)
    depth_values_m = depth_valid / 1000.0
    undistorted_depth = undistorted_homo * depth_values_m.reshape(-1, 1)
    points_3d = undistorted_depth
    # Transform from camera frame to world frame
    R = Rotation.from_rotvec(-rvec).as_matrix()
    # points_3d = (R @ undistorted_depth.T)
    points_3d = (R @ undistorted_depth.T + tvec.reshape(-1, 1)).T

    return points_3d, pixel_coords


def project_image(points_3d, image_shape, K, rvec, tvec, dist=None):
    """
    Project 3D points back to create a depth image with optional distortion.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        image_shape: Output image shape (height, width)
        K: Camera intrinsic matrix (3x3)
        rvec: Rotation vector (camera to world)
        tvec: Translation vector (camera to world)
        dist: Optional distortion coefficients (if None, uses pinhole projection)
    
    Returns:
        image_points: Projected image coordinates (N, 2)
    """
    h, w = image_shape
    
    rvec_inv = -rvec
    tvec_inv = -Rotation.from_rotvec(rvec_inv).as_matrix() @ tvec
    
    if dist is not None:
        # Project with distortion correction
        image_points, _ = cv2.projectPoints(
            points_3d.astype(np.float32), 
            rvec_inv, 
            tvec_inv, 
            K, 
            dist
        )
        return image_points.reshape(-1, 2)
    else:
        # Regular pinhole projection
        # Transform to camera coordinates
        R_inv = Rotation.from_rotvec(rvec_inv).as_matrix()
        points_cam = (R_inv @ points_3d.T + tvec_inv.reshape(-1, 1)).T
        
        # Project to image coordinates: [u, v] = K @ [x/z, y/z, 1]
        points_norm = points_cam / points_cam[:, 2:3]  # Normalize by z
        image_points_homo = (K @ points_norm.T).T
        return image_points_homo[:, :2]


# Utility functions for working with point arrays directly
def unproject_points(image_points, depth_values, K, rvec, tvec, dist=None):
    """
    Utility function: Unproject arrays of image coordinates to 3D world coordinates.
    
    Args:
        image_points: Array of image coordinates (N, 2) where each row is [u, v]
        depth_values: Depth values at pixel locations (N,) or scalar (in mm)
        K: Camera intrinsic matrix (3x3)
        rvec: Rotation vector (camera to world)
        tvec: Translation vector (camera to world)
        dist: Optional distortion coefficients (if None, uses pinhole projection)
    
    Returns:
        points_3d: 3D points in world coordinates (N, 3)
    """
    # Ensure proper input shapes
    image_points = np.asarray(image_points, dtype=np.float32)
    if image_points.ndim == 1:
        image_points = image_points.reshape(1, -1)
    
    depth_values = np.asarray(depth_values)
    if depth_values.ndim == 0:
        depth_values = np.full(image_points.shape[0], depth_values)
    
    if dist is not None:
        # Apply distortion correction
        undistorted = cv2.undistortPoints(image_points, K, dist)
        undistorted = undistorted.reshape(-1, 2)
    else:
        # Regular pinhole projection
        K_inv = np.linalg.inv(K)
        image_points_homo = np.column_stack([image_points, np.ones(len(image_points))])
        undistorted = (K_inv @ image_points_homo.T).T[:, :2]
    
    undistorted_homo = np.concatenate([undistorted, np.ones((undistorted.shape[0], 1))], axis=1)
    
    # Scale by depth and transform
    depth_values_m = depth_values / 1000.0
    undistorted_depth = undistorted_homo * depth_values_m.reshape(-1, 1)
    
    R = Rotation.from_rotvec(rvec).as_matrix()
    points_3d = (R @ undistorted_depth.T + tvec.reshape(-1, 1)).T
    
    return points_3d


def project_points(points_3d, K, rvec, tvec, dist=None):
    """
    Utility function: Project arrays of 3D world coordinates to image coordinates.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        K: Camera intrinsic matrix (3x3)
        rvec: Rotation vector (camera to world)
        tvec: Translation vector (camera to world)
        dist: Optional distortion coefficients (if None, uses pinhole projection)
    
    Returns:
        image_points: Image coordinates (N, 2) where each row is [u, v]
    """
    points_3d = np.asarray(points_3d, dtype=np.float32)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)
    
    # invert rvec and tvec
    rvec_inv = -rvec
    tvec_inv = -Rotation.from_rotvec(rvec_inv).as_matrix() @ tvec
    
    if dist is not None:
        # Project with distortion correction
        image_points, _ = cv2.projectPoints(points_3d, rvec_inv, tvec_inv, K, dist)
        return image_points.reshape(-1, 2)
    else:
        # Regular pinhole projection
        R_inv = Rotation.from_rotvec(rvec_inv).as_matrix()
        points_cam = (R_inv @ points_3d.T + tvec_inv.reshape(-1, 1)).T
        
        # Project to image coordinates
        points_norm = points_cam / points_cam[:, 2:3]
        image_points_homo = (K @ points_norm.T).T
        return image_points_homo[:, :2]