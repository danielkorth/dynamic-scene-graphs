import numpy as np
import configparser
import cv2
import os
import glob
from tqdm import tqdm

def load_poses(path, max_frames=None, subsample=None):
    """Load poses from a CSV file with format: timestamp,tx,ty,tz,rx,ry,rz
    Format: Cam2World
    
    Args:
        path: Path to poses file
        max_frames: Maximum number of frames to load (None for all)
        subsample: Subsample every Nth frame (None for no subsampling)
    
    Returns:
        translations: numpy array of shape (n_frames, 3) with [tx, ty, tz]
        rotations: numpy array of shape (n_frames, 3) with [rx, ry, rz]
    """
    with open(path, "r") as f:
        lines = f.readlines()
    
    translations = []
    rotations = []
    
    for line in lines:
        # Strip whitespace and split by comma
        pose = np.array(line.strip().split(",")).astype(float)
        # Extract translation (indices 1:4) and rotation (indices 4:7)
        translations.append(pose[1:4])
        rotations.append(pose[4:7])
    
    # Convert to numpy arrays
    translations = np.array(translations)
    rotations = np.array(rotations)
    
    # Apply subsampling first (Option A)
    if subsample is not None:
        translations = translations[::subsample]
        rotations = rotations[::subsample]
    
    # Apply max_frames limit after subsampling
    if max_frames is not None:
        translations = translations[:max_frames]
        rotations = rotations[:max_frames]
    
    return translations, -rotations

def load_colmap_poses(path, image_list=None, max_frames=None, subsample=None):
    """
    Load camera poses from a COLMAP images.txt file.
    Each pose is given as: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    Only images with valid poses are included. Optionally, an image_list can be provided to order/align output.

    Args:
        path: Path to COLMAP images.txt file
        image_list: Optional list of image names to align output (missing images get skipped)
        max_frames: Maximum number of frames to load (None for all)
        subsample: Subsample every Nth frame (None for no subsampling)

    Returns:
        translations: numpy array of shape (n_valid, 3)
        rotations: numpy array of shape (n_valid, 3) (rotation vectors)
        names: list of image names with valid poses
    """
    from scipy.spatial.transform import Rotation
    translations = []
    rotations = []
    names = []
    with open(path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        # Parse pose
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        # Convert quaternion to rotation matrix, then to rotation vector
        rotmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        rotvec = Rotation.from_matrix(rotmat).as_rotvec()

        R_wc = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # COLMAP: world-to-cam
        t_wc = np.array([tx, ty, tz])                            # COLMAP: world-to-cam
        # Invert to get cam-to-world:
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        translations.append(t_cw)
        rotations.append(Rotation.from_matrix(R_cw).as_rotvec())
        names.append(name)
        i += 2  # Skip the next line (2D points)
    # If image_list is provided, filter and order output
    if image_list is not None:
        name_to_idx = {n: idx for idx, n in enumerate(names)}
        indices = [name_to_idx[n] for n in image_list if n in name_to_idx]
        translations = [translations[idx] for idx in indices]
        rotations = [rotations[idx] for idx in indices]
        names = [names[idx] for idx in indices]
    # Convert to numpy arrays
    translations = np.array(translations)
    rotations = np.array(rotations)
    # Subsample
    if subsample is not None:
        translations = translations[::subsample]
        rotations = rotations[::subsample]
        names = names[::subsample]
    if max_frames is not None:
        translations = translations[:max_frames]
        rotations = rotations[:max_frames]
        names = names[:max_frames]
    return rotations, translations, names

def load_camera_intrinsics(config_path, camera='left', resolution='2K'):
    """Load camera intrinsics from ZED camera configuration file
    
    Args:
        config_path: Path to the .conf file
        camera: 'left' or 'right' camera
        resolution: '2K', 'FHD', 'HD', or 'VGA'
    
    Returns:
        intrinsics: dict containing fx, fy, cx, cy, k1, k2, k3, p1, p2
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Build section name
    section_name = f"{camera.upper()}_CAM_{resolution.upper()}"
    
    if section_name not in config:
        raise ValueError(f"Section {section_name} not found in config file")
    
    section = config[section_name]
    
    intrinsics = {
        'fx': float(section['fx']),
        'fy': float(section['fy']), 
        'cx': float(section['cx']),
        'cy': float(section['cy']),
        'k1': float(section['k1']),
        'k2': float(section['k2']),
        'k3': float(section['k3']),
        'p1': float(section['p1']),
        'p2': float(section['p2'])
    }
    
    return intrinsics

def get_camera_matrix(intrinsics):
    """Convert intrinsics dict to 3x3 camera matrix K
    
    Args:
        intrinsics: dict with fx, fy, cx, cy
        
    Returns:
        K: 3x3 numpy array camera matrix
    """
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']], 
        [0, 0, 1]
    ])
    return K

def get_distortion_coeffs(intrinsics):
    """Extract distortion coefficients from intrinsics
    
    Args:
        intrinsics: dict with k1, k2, k3, p1, p2
        
    Returns:
        dist_coeffs: numpy array [k1, k2, p1, p2, k3]
    """
    dist_coeffs = np.array([
        intrinsics['k1'],
        intrinsics['k2'], 
        intrinsics['p1'],
        intrinsics['p2'],
        intrinsics['k3']
    ])
    return dist_coeffs

def load_rgb_image(images_dir, frame_number):
    """Load RGB image from ZED camera data
    
    Args:
        images_dir: Path to the images directory
        frame_number: Frame number (int)
        
    Returns:
        rgb_image: numpy array (H, W, 3) in RGB format
    """
    filename = f"left{frame_number:06d}.png"
    image_path = os.path.join(images_dir, filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    
    rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image: {image_path}")
    
    return rgb_image

def load_depth_image(images_dir, frame_number):
    """Load depth image from ZED camera data
    
    Args:
        images_dir: Path to the images directory  
        frame_number: Frame number (int)
        
    Returns:
        depth_image: numpy array (H, W) with depth values
    """
    filename = f"depth{frame_number:06d}.png"
    image_path = os.path.join(images_dir, filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Depth image not found: {image_path}")
    
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        raise ValueError(f"Failed to load depth image: {image_path}")
    
    return depth_image

def load_all_rgb_images(images_dir, max_frames=None, subsample=None):
    """Load all RGB images from ZED camera data directory
    
    Args:
        images_dir: Path to the images directory
        max_frames: Maximum number of frames to load (None for all)
        subsample: Subsample every Nth frame (None for no subsampling)
        
    Returns:
        rgb_images: numpy array of shape (n_frames, H, W, 3) in RGB format
    """
    # Find all left*.png files and sort them
    rgb_pattern = os.path.join(images_dir, "left*.png")
    rgb_files = sorted(glob.glob(rgb_pattern))
    
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {images_dir}")
    
    # Apply subsampling first (Option A)
    if subsample is not None:
        rgb_files = rgb_files[::subsample]
    
    # Apply max_frames limit after subsampling
    if max_frames is not None:
        rgb_files = rgb_files[:max_frames]
    
    rgb_images = []
    
    for rgb_file in tqdm(rgb_files, desc="Loading RGB images"):
        rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        if rgb_image is None:
            print(f"Warning: Failed to load {rgb_file}")
            continue
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb_image)
    
    return np.array(rgb_images)

def load_all_depth_images(images_dir, max_frames=None, subsample=None):
    """Load all depth images from ZED camera data directory
    
    Args:
        images_dir: Path to the images directory
        max_frames: Maximum number of frames to load (None for all)
        subsample: Subsample every Nth frame (None for no subsampling)
        
    Returns:
        depth_images: numpy array of shape (n_frames, H, W) with depth values
    """
    # Find all depth*.png files and sort them
    depth_pattern = os.path.join(images_dir, "depth*.png")
    depth_files = sorted(glob.glob(depth_pattern))
    
    if not depth_files:
        raise FileNotFoundError(f"No depth images found in {images_dir}")
    
    # Apply subsampling first (Option A)
    if subsample is not None:
        depth_files = depth_files[::subsample]
    
    # Apply max_frames limit after subsampling
    if max_frames is not None:
        depth_files = depth_files[:max_frames]
    
    depth_images = []
    
    for depth_file in tqdm(depth_files, desc="Loading depth images"):
        depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Warning: Failed to load {depth_file}")
            continue
        depth_images.append(depth_image)
    
    return np.array(depth_images)