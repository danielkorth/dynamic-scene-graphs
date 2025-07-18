import numpy as np

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