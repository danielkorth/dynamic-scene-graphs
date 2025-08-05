#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import subprocess
import tempfile
import shutil

class VideoMergerConfig:
    """Configuration class for video merger settings."""
    def __init__(self):
        self.fps = 30
        self.colormap = cv2.COLORMAP_JET
        self.colorbar_width = 140  # pixels (increased for better text space)
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.8
        self.text_thickness = 2
        self.text_color = (255, 255, 255)
        self.colorbar_ticks = 5  # number of tick marks
        self.preferred_format = None  # mp4, mov, avi, or None for auto
        self.max_depth_mm = 3000  # maximum depth in millimeters

def detect_depth_format(depth_file):
    """
    Auto-detect depth image format (uint8 vs uint16, channels).
    Returns: (dtype, is_grayscale, min_val, max_val)
    """
    try:
        # Try reading as unchanged first
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not read depth image: {depth_file}")
        
        dtype = depth.dtype
        is_grayscale = len(depth.shape) == 2
        
        # Handle multi-channel depth (like RGBA where RGB contains depth)
        if not is_grayscale and depth.shape[2] >= 3:
            # For RGBA depth, use the R channel (assuming R=G=B)
            depth_data = depth[:, :, 0]
        else:
            depth_data = depth
            
        min_val = np.min(depth_data)
        max_val = np.max(depth_data)
        
        print(f"Detected depth format: {dtype}, grayscale: {is_grayscale}, range: {min_val}-{max_val}")
        return dtype, is_grayscale, min_val, max_val, depth_data
        
    except Exception as e:
        print(f"Error detecting depth format: {e}")
        return None, None, None, None, None

def load_depth_image(depth_file, depth_format_info):
    """Load and process depth image based on detected format."""
    dtype, is_grayscale, _, _, _ = depth_format_info
    
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
        
    # Handle multi-channel depth (extract first channel)
    if not is_grayscale and len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    return depth

def create_colorbar(height, width, min_depth, max_depth, colormap, config, is_millimeters=False):
    """Create a constant colorbar for configurable depth range."""
    # Create gradient that exactly matches depth image normalization
    # This ensures colorbar and depth images use identical color mapping
    # Gradient: 0 (top) to 255 (bottom) to match depth mapping
    gradient = np.linspace(0, 255, height).astype(np.uint8)
    gradient = np.repeat(gradient[:, np.newaxis], width//3, axis=1)
    
    # Apply the same colormap as depth images
    colorbar = cv2.applyColorMap(gradient, colormap)
    
    # Expand to full width
    colorbar_full = np.zeros((height, width, 3), dtype=np.uint8)
    colorbar_full[:, :width//3] = colorbar
    
    # Calculate tick values based on max depth
    max_depth_val = config.max_depth_mm
    if max_depth_val <= 1000:
        step = 200  # Every 200mm for ranges up to 1m
    elif max_depth_val <= 3000:
        step = 500  # Every 500mm for ranges up to 3m
    elif max_depth_val <= 5000:
        step = 1000  # Every 1000mm for ranges up to 5m
    else:
        step = 2000  # Every 2000mm for larger ranges
    
    tick_values = list(range(0, int(max_depth_val) + 1, step))
    if tick_values[-1] != max_depth_val:
        tick_values.append(int(max_depth_val))
    
    # Add tick marks and labels
    for depth_value in tick_values:
        # Calculate y position: 0mm at top (value 0), max_depth at bottom (value 255)
        # This matches the depth image where 0mm maps to 0, max_depth maps to 255
        normalized_pos = depth_value / float(config.max_depth_mm)  # Use configurable range
        y = int(normalized_pos * height)
        y = max(0, min(height - 1, y))  # Clamp to valid range
        
        # Draw tick mark
        cv2.line(colorbar_full, (width//3, y), (width//3 + 15, y), (255, 255, 255), 3)
        
        # Format label for millimeters (always show meters for 1000+ values)
        if depth_value >= 1000:
            label = f"{depth_value/1000:.0f}m"
        else:
            label = f"{int(depth_value)}mm"
            
        # Position text
        font_scale = config.text_scale * 1.1
        text_thickness = 2
        text_size = cv2.getTextSize(label, config.text_font, font_scale, text_thickness)[0]
        text_x = width//3 + 25
        text_y = y + text_size[1]//2
        
        # Ensure text doesn't go off the image
        if text_x + text_size[0] > width:
            text_x = width - text_size[0] - 5
        
        cv2.putText(colorbar_full, label, (text_x, text_y), 
                   config.text_font, font_scale, 
                   config.text_color, text_thickness)
    
    return colorbar_full

def colorize_depth_image(depth_image, min_depth, max_depth, colormap, max_depth_mm=3000):
    """Colorize depth image with configurable maximum range."""
    # Set configurable range: 0 to max_depth_mm
    max_display_depth = float(max_depth_mm)
    min_display_depth = 0.0
    
    # Clip depth values to specified range
    depth_clipped = np.clip(depth_image.astype(np.float32), min_display_depth, max_display_depth)
    
    # Normalize to 0-255 range for colormap
    depth_normalized = ((depth_clipped - min_display_depth) / 
                       (max_display_depth - min_display_depth) * 255)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # Apply colormap
    colored_depth = cv2.applyColorMap(depth_normalized, colormap)
    
    return colored_depth

def add_text_with_background(image, text, position, config):
    """Add text with dark background for better visibility."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, config.text_font, config.text_scale, config.text_thickness)
    
    x, y = position
    cv2.rectangle(image, (x, y - text_height - baseline), 
                 (x + text_width, y + baseline), (0, 0, 0), -1)
    
    cv2.putText(image, text, position, config.text_font, 
               config.text_scale, config.text_color, config.text_thickness)

def get_image_files(input_dir):
    """Get sorted lists of RGB and depth image files."""
    rgb_files = sorted(glob.glob(os.path.join(input_dir, "left*.png")))
    depth_files = sorted(glob.glob(os.path.join(input_dir, "depth*.png")))
    
    return rgb_files, depth_files

def calculate_global_depth_range(depth_files, depth_format_info, max_depth_mm, sample_size=50):
    """Return fixed depth range from 0 to specified maximum."""
    print(f"Using fixed depth range: 0-{max_depth_mm}mm ({max_depth_mm/1000:.1f} meters)")
    
    # Return fixed range instead of calculating from data
    global_min = 0.0
    global_max = float(max_depth_mm)
    
    print(f"Fixed depth range: {global_min} - {global_max}mm")
    return global_min, global_max

def create_video_with_ffmpeg(temp_dir, output_path, fps, preferred_format=None):
    """Create video using ffmpeg from saved frames."""
    
    # Determine output format and codec
    if preferred_format:
        if preferred_format.lower() == 'mp4':
            extension = '.mp4'
            codec_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        elif preferred_format.lower() == 'avi':
            extension = '.avi'
            codec_args = ['-c:v', 'libxvid']
        elif preferred_format.lower() == 'mov':
            extension = '.mov'
            codec_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        else:
            extension = '.mp4'
            codec_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    else:
        extension = '.mp4'
        codec_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    
    # Ensure output has correct extension
    base_name = os.path.splitext(output_path)[0]
    output_file = base_name + extension
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%06d.png'),
        '-r', str(fps),  # output framerate
    ] + codec_args + [output_file]
    
    try:
        print(f"Creating video with ffmpeg: {output_file}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Video created successfully: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise RuntimeError(f"FFmpeg failed to create video: {e}")

def create_unified_video(input_dir, output_video_path, config=None):
    """
    Create unified video with RGB | Depth | Colorbar layout.
    Auto-detects uint8/uint16 depth format.
    """
    if config is None:
        config = VideoMergerConfig()
    
    # Get image files
    rgb_files, depth_files = get_image_files(input_dir)
    
    if not rgb_files or not depth_files:
        print("Error: No RGB or depth images found!")
        print(f"RGB files: {len(rgb_files)}, Depth files: {len(depth_files)}")
        return False
    
    if len(rgb_files) != len(depth_files):
        print(f"Warning: Mismatched file counts - RGB: {len(rgb_files)}, Depth: {len(depth_files)}")
        min_files = min(len(rgb_files), len(depth_files))
        rgb_files = rgb_files[:min_files]
        depth_files = depth_files[:min_files]
    
    print(f"Processing {len(rgb_files)} image pairs")
    
    # Detect depth format from first image
    depth_format_info = detect_depth_format(depth_files[0])
    if depth_format_info[0] is None:
        print("Error: Could not detect depth format!")
        return False
    
    # Read first images to get dimensions
    rgb_img = cv2.imread(rgb_files[0])
    depth_img = load_depth_image(depth_files[0], depth_format_info)
    
    if rgb_img is None or depth_img is None:
        print("Error: Could not read first images!")
        return False
    
    height, width = rgb_img.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Calculate global depth range
    global_min_depth, global_max_depth = calculate_global_depth_range(
        depth_files, depth_format_info, config.max_depth_mm)
    
    # Create temporary directory for frames
    total_width = width * 2 + config.colorbar_width
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process frames
        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(rgb_files)}")
            
            # Read images
            rgb_img = cv2.imread(rgb_file)
            depth_img = load_depth_image(depth_file, depth_format_info)
            
            if rgb_img is None or depth_img is None:
                print(f"Error reading frame {i+1}, skipping...")
                continue
            
            # Colorize depth with configurable clipping
            colored_depth = colorize_depth_image(
                depth_img, global_min_depth, global_max_depth, config.colormap, config.max_depth_mm)
            
            # Create colorbar for configurable range
            colorbar = create_colorbar(
                height, config.colorbar_width, global_min_depth, global_max_depth, 
                config.colormap, config, True)  # Always use millimeter formatting
            
            # Create three-panel layout: RGB | Depth | Colorbar
            combined = np.zeros((height, total_width, 3), dtype=np.uint8)
            combined[:, :width] = rgb_img
            combined[:, width:2*width] = colored_depth
            combined[:, 2*width:] = colorbar
            
            # Add labels with dynamic range
            max_depth_meters = config.max_depth_mm / 1000.0
            add_text_with_background(combined, "RGB", (10, 30), config)
            add_text_with_background(combined, f"Depth (0-{max_depth_meters:.1f}m)", (width + 10, 30), config)
            add_text_with_background(combined, f"Frame: {i+1:04d}", (10, height - 25), config)
            
            # Add depth info
            depth_info = f"Range: 0-{config.max_depth_mm}mm"
            add_text_with_background(combined, depth_info, (width + 10, height - 25), config)
            
            # Save frame as PNG
            frame_path = os.path.join(temp_dir, f"frame_{i+1:06d}.png")
            cv2.imwrite(frame_path, combined)
        
        # Create video using ffmpeg
        output_file = create_video_with_ffmpeg(
            temp_dir, output_video_path, config.fps, 
            getattr(config, 'preferred_format', None))
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Create unified RGB+Depth+Colorbar video')
    parser.add_argument('input_dir', help='Directory containing left*.png and depth*.png files')
    parser.add_argument('-o', '--output', default='unified_video.mp4', 
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate')
    parser.add_argument('--colorbar-width', type=int, default=140, 
                       help='Colorbar width in pixels')
    parser.add_argument('--max-depth', type=int, default=3000, 
                       help='Maximum depth in millimeters (default: 3000mm = 3m)')
    parser.add_argument('--format', choices=['mp4', 'mov', 'avi'], 
                       help='Preferred video format (mp4, mov, or avi). Will try fallback formats if preferred fails.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found!")
        return 1
    
    # Setup configuration
    config = VideoMergerConfig()
    config.fps = args.fps
    config.colorbar_width = args.colorbar_width
    config.max_depth_mm = args.max_depth
    config.preferred_format = args.format
    
    # Create video
    success = create_unified_video(args.input_dir, args.output, config)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 