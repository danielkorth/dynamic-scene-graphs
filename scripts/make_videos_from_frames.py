import argparse
import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import hashlib


def natural_frame_sort(file_names: List[str]) -> List[str]:
    """Sort frame file names by the integer suffix if present, else lexicographically.

    Examples:
    - depth000001.png < depth000010.png
    - left000001.png < left000010.png
    """
    def extract_key(name: str) -> Tuple[str, int]:
        base = os.path.basename(name)
        match = re.search(r"(\d+)(?=\.[^.]+$)", base)
        if match:
            return (base[: match.start()], int(match.group(1)))
        return (base, -1)

    return sorted(file_names, key=extract_key)


def get_frame_hash(frame_path: str) -> str:
    """Generate a hash for a frame to detect duplicates."""
    img = cv2.imread(frame_path)
    if img is None:
        return ""
    # Convert to grayscale and resize for consistent hashing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    return hashlib.md5(img_resized.tobytes()).hexdigest()


def detect_overlapping_frames(folder_paths: List[str], expected_overlap: int = 1) -> List[Tuple[int, int]]:
    """Detect overlapping frames between consecutive folders using known overlap pattern.

    Args:
        folder_paths: List of folder paths to analyze
        expected_overlap: Expected number of overlapping frames (default: 1 based on sam2_reinit.yaml)

    Returns a list of tuples (folder_idx, frame_idx_to_skip) indicating
    which frames to skip due to overlap with previous folder.
    """
    overlaps = []

    # For sam2_reinit.py with overlap=1, we expect the first frame of each folder
    # (except the first folder) to overlap with the last frame of the previous folder
    for i in range(1, len(folder_paths)):
        print(f"Detected expected overlap: {os.path.basename(folder_paths[i])} first {expected_overlap} frame(s) overlap with {os.path.basename(folder_paths[i-1])}")
        overlaps.append((i, 0))  # Skip first frame of current folder

    return overlaps


def find_frames(input_dir: str) -> Tuple[List[str], List[str]]:
    """Find depth and rgb frame paths in the folder.

    Depth frames are expected to start with 'depth'.
    RGB frames are searched by common prefixes: 'left', 'rgb', 'color', 'image', 'img'.
    Only image files are considered.
    """
    supported_exts = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]

    depth_files = [f for f in all_files if os.path.basename(f).lower().startswith("depth")]

    rgb_prefix_candidates = ["left", "rgb", "color", "image", "img"]
    rgb_files: List[str] = []
    for prefix in rgb_prefix_candidates:
        rgb_files = [
            f
            for f in all_files
            if os.path.basename(f).lower().startswith(prefix)
        ]
        if rgb_files:
            break
    # Fallback: take all non-depth images
    if not rgb_files:
        rgb_files = [f for f in all_files if f not in depth_files]

    depth_paths = [os.path.join(input_dir, f) for f in depth_files]
    rgb_paths = [os.path.join(input_dir, f) for f in rgb_files]

    depth_paths = natural_frame_sort(depth_paths)
    rgb_paths = natural_frame_sort(rgb_paths)

    return depth_paths, rgb_paths


def ensure_writer(path: str, fps: int, frame_shape_hw: Tuple[int, int], is_color: bool) -> cv2.VideoWriter:
    height, width = frame_shape_hw
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def to_bgr8(image: np.ndarray) -> np.ndarray:
    """Convert an image to 8-bit 3-channel BGR for video writing.

    - If already uint8 3-channel, return as-is.
    - If uint8 single-channel, convert to BGR.
    - If float or uint16 single-channel, normalize to 0-255 and apply a colormap for visibility.
    - If float or uint16 3-channel, normalize per-channel and convert to uint8.
    """
    if image.dtype == np.uint8:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
    # Single-channel depth-like data
    if image.ndim == 2:
        # Robust normalization: ignore zeros (often invalid depth)
        valid = image > 0
        if np.any(valid):
            vmin = float(np.percentile(image[valid], 1.0))
            vmax = float(np.percentile(image[valid], 99.0))
            if vmax <= vmin:
                vmax = float(image[valid].max())
                vmin = float(image[valid].min())
        else:
            vmin, vmax = 0.0, 1.0
        img = image.astype(np.float32)
        img = np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        img8 = (img * 255.0 + 0.5).astype(np.uint8)
        return cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    # Three-channel float/uint16
    if image.ndim == 3 and image.shape[2] == 3:
        img = image.astype(np.float32)
        img = np.clip(img, 0, None)
        # Normalize per-channel
        for c in range(3):
            ch = img[:, :, c]
            if ch.max() > 0:
                ch /= max(ch.max(), 1e-6)
                img[:, :, c] = ch
        return (img * 255.0 + 0.5).astype(np.uint8)
    # Fallback to grayscale conversion
    img = image.astype(np.float32)
    img -= img.min()
    rng = img.max()
    if rng > 0:
        img /= rng
    img8 = (img * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)


def write_video_from_frames(
    frames: List[str],
    output_path: str,
    fps: int,
    treat_as_depth: bool = False,
) -> Optional[str]:
    if not frames:
        return None

    # Read first frame to initialize writer
    first = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")

    if treat_as_depth:
        first_for_shape = to_bgr8(first)
    else:
        # If RGB is loaded as BGR, ensure it's 3-channel uint8 BGR
        if first.ndim == 2:
            first_for_shape = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
        elif first.ndim == 3 and first.shape[2] == 3:
            # Assume BGR8 already
            first_for_shape = first.astype(np.uint8)
        else:
            first_for_shape = to_bgr8(first)

    writer = ensure_writer(output_path, fps, (first_for_shape.shape[0], first_for_shape.shape[1]), is_color=True)
    writer.write(first_for_shape)

    # Process remaining frames
    target_size = (first_for_shape.shape[1], first_for_shape.shape[0])
    for path in frames[1:]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if treat_as_depth:
            frame = to_bgr8(img)
        else:
            if img.ndim == 2:
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
                frame = img
            else:
                frame = to_bgr8(img)
        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    return output_path


def collect_visualization_folders(input_dir: str) -> List[str]:
    """Collect all visualization_* folders and return them in order."""
    vis_folders = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and item.startswith("visualizations_"):
            vis_folders.append(item_path)

    # Sort by the numeric suffix
    def extract_vis_number(folder_path: str) -> int:
        folder_name = os.path.basename(folder_path)
        match = re.search(r"visualizations_(\d+)", folder_name)
        return int(match.group(1)) if match else 0

    return sorted(vis_folders, key=extract_vis_number)


def create_video_from_multiple_folders(folder_paths: List[str], output_path: str, fps: int, treat_as_depth: bool = False) -> Optional[str]:
    """Create a video from multiple folders, handling overlaps between consecutive folders."""
    if not folder_paths:
        return None

    # Detect overlapping frames (using expected overlap from sam2_reinit.yaml config)
    overlaps = detect_overlapping_frames(folder_paths, expected_overlap=1)
    overlap_dict = dict(overlaps)  # folder_idx -> frame_idx_to_skip

    all_frames = []
    global_frame_count = 0

    for folder_idx, folder_path in enumerate(folder_paths):
        depth_frames, rgb_frames = find_frames(folder_path)
        frames_to_use = rgb_frames if not treat_as_depth else depth_frames

        if not frames_to_use:
            continue

        # Skip overlapping frames if this folder has overlaps
        start_idx = overlap_dict.get(folder_idx, -1) + 1
        frames_to_add = frames_to_use[start_idx:]

        if frames_to_add:
            all_frames.extend(frames_to_add)
            print(f"Added {len(frames_to_add)} frames from {os.path.basename(folder_path)} (skipped {start_idx} overlapping frames)")
            global_frame_count += len(frames_to_add)

    if not all_frames:
        return None

    print(f"Total frames to process: {global_frame_count}")
    return write_video_from_frames(all_frames, output_path, fps, treat_as_depth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RGB and depth videos from frame folders.")
    parser.add_argument("input_dir", type=str, help="Path to folder containing visualization folders")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output videos")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write videos (defaults to input_dir)",
    )
    parser.add_argument(
        "--rgb-name",
        type=str,
        default="rgb_video.mp4",
        help="Filename for the RGB video",
    )
    parser.add_argument(
        "--depth-name",
        type=str,
        default="depth_video.mp4",
        help="Filename for the depth video",
    )
    parser.add_argument(
        "--single-folder",
        action="store_true",
        help="Process as single folder instead of multiple visualization folders",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else input_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.single_folder:
        # Original behavior for single folder processing
        depth_frames, rgb_frames = find_frames(input_dir)

        if not depth_frames:
            print("No depth frames found (expected files like depth000000.png). Skipping depth video.")
        if not rgb_frames:
            print("No RGB frames found (looked for prefixes: left, rgb, color, image, img). Skipping RGB video.")

        if rgb_frames:
            rgb_out = os.path.join(out_dir, args.rgb_name)
            path = write_video_from_frames(rgb_frames, rgb_out, fps=args.fps, treat_as_depth=False)
            if path:
                print(f"Wrote RGB video: {path}")

        if depth_frames:
            depth_out = os.path.join(out_dir, args.depth_name)
            path = write_video_from_frames(depth_frames, depth_out, fps=args.fps, treat_as_depth=True)
            if path:
                print(f"Wrote depth video: {path}")
    else:
        # New behavior for multiple visualization folders
        vis_folders = collect_visualization_folders(input_dir)
        if not vis_folders:
            print("No visualization_* folders found. Use --single-folder if you want to process a single folder.")
            return

        print(f"Found {len(vis_folders)} visualization folders: {[os.path.basename(f) for f in vis_folders]}")

        # Create RGB video
        rgb_out = os.path.join(out_dir, args.rgb_name)
        path = create_video_from_multiple_folders(vis_folders, rgb_out, args.fps, treat_as_depth=False)
        if path:
            print(f"Wrote RGB video: {path}")

        # Create depth video
        depth_out = os.path.join(out_dir, args.depth_name)
        path = create_video_from_multiple_folders(vis_folders, depth_out, args.fps, treat_as_depth=True)
        if path:
            print(f"Wrote depth video: {path}")


if __name__ == "__main__":
    main()


