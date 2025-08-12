#!/usr/bin/env python3
"""
Script to visualize the saved Open3D textured pointclouds from the final_reconstructions folder.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import os

def visualize_reconstructions(reconstructions_folder: str, max_points_per_cloud: int = 10000):
    """
    Visualize all saved pointclouds from the final_reconstructions folder.
    
    Args:
        reconstructions_folder: Path to the final_reconstructions folder
        max_points_per_cloud: Maximum number of points to display per cloud (for performance)
    """
    reconstructions_path = Path(reconstructions_folder)
    
    if not reconstructions_path.exists():
        print(f"Error: Reconstructions folder '{reconstructions_folder}' does not exist!")
        return
    
    # Find all PLY files
    ply_files = list(reconstructions_path.glob("*.ply"))
    
    if not ply_files:
        print(f"No PLY files found in '{reconstructions_folder}'")
        return
    
    print(f"Found {len(ply_files)} pointcloud files:")
    for ply_file in ply_files:
        print(f"  - {ply_file.name}")
    
    # Load and prepare pointclouds
    geometries = []
    
    selected_files = ply_files
    for i, ply_file in enumerate(selected_files):
        print(f"Loading {ply_file.name}...")
        
        try:
            # Load pointcloud
            pcd = o3d.io.read_point_cloud(str(ply_file))
            
            if len(pcd.points) == 0:
                print(f"  Warning: {ply_file.name} has no points, skipping...")
                continue
            
            # Downsample if too many points
            if len(pcd.points) > max_points_per_cloud:
                print(f"  Downsampling from {len(pcd.points)} to {max_points_per_cloud} points...")
                pcd = pcd.voxel_down_sample(voxel_size=0.01)
                # If still too many points, use uniform downsampling
                if len(pcd.points) > max_points_per_cloud:
                    indices = np.random.choice(len(pcd.points), max_points_per_cloud, replace=False)
                    pcd = pcd.select_by_index(indices)
            
            # Generate unique color for each object if no colors exist
            if len(pcd.colors) == 0:
                # Generate a distinct color based on object index
                hue = (i * 137.508) % 360  # Golden angle for good color distribution
                rgb = hsv_to_rgb(hue, 0.7, 0.9)
                pcd.paint_uniform_color(rgb)
                print(f"  Applied generated color: {rgb}")
            
            # Add to visualization list
            geometries.append(pcd)
            
            # Add label
            label = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            label.translate(np.mean(pcd.points, axis=0))
            geometries.append(label)
            
            print(f"  Loaded {len(pcd.points)} points")
            
        except Exception as e:
            print(f"  Error loading {ply_file.name}: {e}")
            continue
    
    if not geometries:
        print("No valid pointclouds loaded!")
        return
    
    print(f"\nVisualizing {len(geometries)} geometries...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Shift + Mouse: Pan view")
    print("  - Mouse wheel: Zoom")
    print("  - H: Show/hide coordinate frames")
    print("  - Q: Quit")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Scene Graph Reconstructions",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=True
    )

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB color."""
    h = h / 360.0
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if i % 6 == 0:
        return [v, t, p]
    elif i % 6 == 1:
        return [q, v, p]
    elif i % 6 == 2:
        return [p, v, t]
    elif i % 6 == 3:
        return [p, q, v]
    elif i % 6 == 4:
        return [t, p, v]
    else:
        return [v, p, q]

def main():
    parser = argparse.ArgumentParser(description="Visualize Open3D reconstructions")
    parser.add_argument(
        "reconstructions_folder", 
        help="Path to the final_reconstructions folder"
    )
    parser.add_argument(
        "--max-points", 
        type=int, 
        default=10000,
        help="Maximum points per cloud for performance (default: 10000)"
    )
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.reconstructions_folder):
        print(f"Error: Folder '{args.reconstructions_folder}' does not exist!")
        print("Please run the main script first to generate reconstructions.")
        return
    
    visualize_reconstructions(args.reconstructions_folder, args.max_points)

if __name__ == "__main__":
    main() 