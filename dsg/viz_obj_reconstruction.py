#!/usr/bin/env python3
"""
Script to visualize the saved Open3D textured pointclouds from the final_reconstructions folder.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import os

def reconstruct_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, poisson_depth: int = 9) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a triangle mesh from a point cloud using Poisson surface reconstruction.

    The function estimates and orients normals if missing, then runs Poisson,
    prunes low-density vertices, and cleans the resulting mesh.

    Args:
        pcd: Open3D point cloud
        poisson_depth: Octree depth for Poisson (higher = more detail, slower)

    Returns:
        Cleaned Open3D triangle mesh
    """
    # Ensure normals exist
    if not pcd.has_normals():
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        normal_radius = max(diag * 0.01, 1e-6)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(k=50)
        except Exception:
            # Fallback orientation if consistent orientation fails
            pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))

    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     pcd, depth=poisson_depth
    # )
    # Remove low-density vertices (trim floating artifacts)
    # densities = np.asarray(densities)
    # if densities.size == len(mesh.vertices) and densities.size > 0:
    #     density_threshold = np.quantile(densities, 0.01)
    #     to_remove_mask = densities < density_threshold
    #     mesh.remove_vertices_by_mask(to_remove_mask)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.02)

    # # Clean mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    # mesh.compute_vertex_normals()
    return mesh

def visualize_reconstructions(reconstructions_folder: str, max_points_per_cloud: int = 10000, viz_meshes: bool = True):
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
    
    # selected_files = [ply_files[9]] # 9-umbrella
    # selected_files = [ply_files[0]] # 0-lamp
    # selected_files = [ply_files[19]]# 19-notebook

    selected_files = [ply_files[28], ply_files[13]]
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
            rgb = None
            if len(pcd.colors) == 0:
                hue = (i * 137.508) % 360  # Golden angle for good color distribution
                rgb = hsv_to_rgb(hue, 0.7, 0.9)
                pcd.paint_uniform_color(rgb)
                print(f"  Applied generated color: {rgb}")

            # Reconstruct mesh from point cloud and visualize that instead
            if viz_meshes:
                try:
                    mesh = reconstruct_mesh_from_pointcloud(pcd)
                    if (rgb is not None) and (not mesh.has_vertex_colors()):
                        mesh.paint_uniform_color(rgb)
                    geometries.append(mesh)
                    print(f"  Reconstructed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                except Exception as mesh_err:
                    print(f"  Mesh reconstruction failed, showing points instead: {mesh_err}")
                    geometries.append(pcd)
            else:
                geometries.append(pcd)
            
            # # Add label
            # label = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # label.translate(np.mean(pcd.points, axis=0))
            # geometries.append(label)
            
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