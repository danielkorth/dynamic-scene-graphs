#!/usr/bin/env python3
"""
Script to visualize the saved Open3D textured pointclouds from the final_reconstructions folder.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import os
import hydra
from omegaconf import DictConfig

def reconstruct_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, alpha_shape_alpha: float = 0.02) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a triangle mesh from a point cloud using Poisson surface reconstruction.

    The function estimates and orients normals if missing, then runs Poisson,
    prunes low-density vertices, and cleans the resulting mesh.

    Args:
        pcd: Open3D point cloud
        poisson_depth: Octree depth for Poisson (higher = more detail, slower)
        alpha_shape_alpha: Alpha value for alpha shape reconstruction

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

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha_shape_alpha)

    # # Clean mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    # mesh.compute_vertex_normals()
    return mesh

def visualize_reconstructions(cfg: DictConfig):
    """
    Visualize all saved pointclouds from the final_reconstructions folder.
    
    Args:
        cfg: Hydra configuration object
    """
    reconstructions_path = Path(cfg.reconstructions_folder)
    
    if not reconstructions_path.exists():
        print(f"Error: Reconstructions folder '{cfg.reconstructions_folder}' does not exist!")
        return
    
    # Find all PLY files
    ply_files = list(reconstructions_path.glob("*.ply"))
    
    if not ply_files:
        print(f"No PLY files found in '{cfg.reconstructions_folder}'")
        return
    
    print(f"Found {len(ply_files)} pointcloud files:")
    for ply_file in ply_files:
        print(f"  - {ply_file.name}")
    
    # Load and prepare pointclouds
    geometries = []
    
    # Use selected objects from config if specified, otherwise use all files
    if hasattr(cfg, 'selected_objects') and cfg.selected_objects:
        selected_files = [ply_files[i] for i in cfg.selected_objects if i < len(ply_files)]
        if not selected_files:
            print(f"Warning: No valid object indices in selected_objects: {cfg.selected_objects}")
            selected_files = ply_files
    else:
        # Default selection for demonstration (can be overridden in config)
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
            if len(pcd.points) > cfg.max_points_per_cloud:
                print(f"  Downsampling from {len(pcd.points)} to {cfg.max_points_per_cloud} points...")
                pcd = pcd.voxel_down_sample(voxel_size=0.01)
                # If still too many points, use uniform downsampling
                if len(pcd.points) > cfg.max_points_per_cloud:
                    indices = np.random.choice(len(pcd.points), cfg.max_points_per_cloud, replace=False)
                    pcd = pcd.select_by_index(indices)
            
            # Generate unique color for each object if no colors exist
            rgb = None
            if len(pcd.colors) == 0:
                hue = (i * 137.508) % 360  # Golden angle for good color distribution
                rgb = hsv_to_rgb(hue, 0.7, 0.9)
                pcd.paint_uniform_color(rgb)
                print(f"  Applied generated color: {rgb}")

            # Reconstruct mesh from point cloud and visualize that instead
            if cfg.viz_meshes:
                try:
                    mesh = reconstruct_mesh_from_pointcloud(
                        pcd, 
                        alpha_shape_alpha=getattr(cfg, 'alpha_shape_alpha', 0.02)
                    )
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
        width=getattr(cfg, 'window_width', 1200),
        height=getattr(cfg, 'window_height', 800),
        point_show_normal=getattr(cfg, 'point_show_normal', False),
        mesh_show_wireframe=getattr(cfg, 'mesh_show_wireframe', False),
        mesh_show_back_face=getattr(cfg, 'mesh_show_back_face', True)
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

@hydra.main(config_path="../configs", config_name="video_tracking")
def main(cfg: DictConfig):
    print(f"Configuration: {cfg}")
    
    # Check if folder exists
    if not os.path.exists(cfg.reconstructions_folder):
        print(f"Error: Folder '{cfg.reconstructions_folder}' does not exist!")
        print("Please run the main script first to generate reconstructions.")
        return
    
    visualize_reconstructions(cfg)

if __name__ == "__main__":
    main() 