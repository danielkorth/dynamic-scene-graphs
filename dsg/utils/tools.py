import numpy as np

# --------------------------------------------
# Utility: generate a consistent color per object ID
# --------------------------------------------
def get_color_for_id(obj_id):
    golden_angle = 137.508
    hue = (obj_id * golden_angle) % 360 / 360.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.90)
    return [r, g, b]

    
def center_crop(img, crop_px):
    h, w = img.shape[:2]
    start_x = crop_px
    start_y = crop_px
    end_x = w - crop_px
    end_y = h - crop_px
    return img[start_y:end_y, start_x:end_x]

def get_bounding_box(mask):
    """
    Returns the bounding box coordinates (y_min, x_min, y_max, x_max) from a 2D binary mask.

    Parameters:
        mask (np.ndarray): 2D NumPy array of shape (H, W), with boolean or 0/1 values.

    Returns:
        tuple: (y_min, x_min, y_max, x_max) or None if mask is empty.
    """
    if not np.any(mask):
        return None  # No non-zero elements

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add +1 to max coords if you want an exclusive bounding box
    return np.array([y_min, x_min, y_max + 1, x_max + 1])

def sample_points_in_mask(mask, num_points, method='random'):
    """
    Samples points inside a binary mask.

    Parameters:
        mask (np.ndarray): 2D boolean or binary array (H, W).
        num_points (int): Number of points to sample.
        method (str): 'random' or 'uniform' sampling strategy.

    Returns:
        np.ndarray: Array of shape (N, 2) with (y, x) coordinates of sampled points.
    """
    if not np.any(mask):
        raise ValueError("Mask is empty. No points to sample.")

    mask_indices = np.argwhere(mask)  # Shape: (num_valid_pixels, 2)

    if method == 'random':
        if len(mask_indices) < num_points:
            raise ValueError("Not enough valid points in mask to sample without replacement.")
        sampled_idx = np.random.choice(len(mask_indices), size=num_points, replace=False)
        return mask_indices[sampled_idx]

    elif method == 'uniform':
        h, w = mask.shape
        bbox = get_bounding_box(mask)
        if bbox is None:
            raise ValueError("Cannot find bounding box in empty mask.")

        y_min, x_min, y_max, x_max = bbox
        grid_size = int(np.ceil(np.sqrt(num_points)))

        y_lin = np.linspace(y_min, y_max - 1, grid_size).astype(int)
        x_lin = np.linspace(x_min, x_max - 1, grid_size).astype(int)

        grid_points = np.array(np.meshgrid(y_lin, x_lin)).reshape(2, -1).T  # (N, 2)

        # Filter points inside the mask
        valid_points = np.array([pt for pt in grid_points if mask[pt[0], pt[1]]])

        if len(valid_points) < num_points:
            # Fallback to random sampling for remaining points
            extra_needed = num_points - len(valid_points)
            remaining_points = np.setdiff1d(mask_indices.view([('', mask_indices.dtype)]*2), 
                                            valid_points.view([('', valid_points.dtype)]*2))
            remaining_points = remaining_points.view(mask_indices.dtype).reshape(-1, 2)
            if len(remaining_points) >= extra_needed:
                extra_samples = remaining_points[np.random.choice(len(remaining_points), extra_needed, replace=False)]
                return np.vstack([valid_points, extra_samples])
            else:
                return valid_points  # Return as many as possible
        else:
            return valid_points[:num_points]

    else:
        raise ValueError(f"Unknown sampling method '{method}'. Use 'random' or 'uniform'.")
    
def mask_union_and_coverage(masks):
    if isinstance(masks, list) and isinstance(masks[0], dict):
        all_seg = np.stack([m['segmentation'] for m in masks], axis=0)  # N×H×W
    elif isinstance(masks, list) and isinstance(masks[0], np.ndarray):
        all_seg = np.stack(masks, axis=0)
    elif isinstance(masks, np.ndarray):
        pass
    else:
        raise(Exception("Bad Mask Formatting"))
    
    H, W = all_seg[0].squeeze().shape
    # 1) compute union mask
    union = np.any(all_seg, axis=0)                                 # H×W
    coverage = union.sum() / float(H * W)

    return union.squeeze(), coverage