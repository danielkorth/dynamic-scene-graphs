from dsg.scenegraph.node import Node, Edge
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from dsg.utils.tools import get_color_for_id
from scipy.spatial.transform import Rotation
from dsg.utils.cv2_utils import unproject_image
import cv2

def process_frame_with_representation(rgb, depth, tvec, rvec, obj_points, K, graph, cfg, use_tsdf=False, debug_mask=False):
    """
    Process a frame and update the scene graph with either TSDF or point cloud representation.
    
    Args:
        rgb: RGB image
        depth: Depth image
        tvec: Translation vector
        rvec: Rotation vector
        obj_points: Object points dictionary
        K: Camera intrinsic matrix
        graph: Scene graph
        cfg: Configuration
        use_tsdf: Whether to use TSDF representation
        use_dynamic_tracking: Whether to use dynamic tracking for moving objects
    """
    # Create camera pose matrix for TSDF integration
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    camera_pose[:3, 3] = tvec
    
    for obj_id, obj_point in obj_points.items():
        if obj_point['mask'].sum() <= 0:
            continue
        erosion_kernel = 13 if cfg.depth_source is not None and "moge" in cfg.depth_source else 5
        points_3d, pixel_coords = unproject_image(depth, K, -rvec, tvec, mask=obj_point['mask'], dist=None, mask_erode_kernel=erosion_kernel)
        # Convert pixel coordinates to integers for array indexing
        pixel_coords_int = pixel_coords.astype(np.int32)
        rgb_points = rgb[pixel_coords_int[:, 1], pixel_coords_int[:, 0]] / 255.0
        centroid = np.mean(points_3d, axis=0)

        if len(points_3d) > 0:
            if f"obj_{obj_id}" not in graph:
                # Generate unique color for this object
                color = np.array(get_color_for_id(obj_id))
                node = Node(f"obj_{obj_id}", centroid, color=color, pct=points_3d, rgb_points=rgb_points,
                        use_tsdf=use_tsdf)
                graph.add_node(node)
                
                # If using TSDF, integrate the first frame
                if use_tsdf:
                    node.integrate_frame_to_tsdf(depth, rgb, obj_point['mask'], K, camera_pose)

            else:
                node = graph.nodes[f'obj_{obj_id}']
                
                if use_tsdf:
                    # Integrate frame into TSDF
                    node.integrate_frame_to_tsdf(depth, rgb, obj_point['mask'], K, camera_pose)
                    points_3d, colors = node.tsdf.extract_point_cloud() 
                else:
                    node.integrate_pointcloud(points_3d, rgb_points, cfg.accumulate_points, 
                                            camera_intrinsics=K, camera_pose=camera_pose, 
                                            current_mask=obj_point['mask'], depth_image=depth, obj_id=obj_id)
            

class SceneGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
    
    def __contains__(self, node_name: str):
        return node_name in self.nodes

    def log_rerun(self, show_pct: bool = False, max_points: int = 5000, edge_threshold: float = 1.0):
        """Log scene graph to rerun"""
        # Create edges between nodes that are closer than edge_threshold meters
        edge_strips = []
        edge_colors = []

        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]
                distance = np.linalg.norm(node1.centroid - node2.centroid)
                if distance < edge_threshold:
                    edge_strips.append([node1.centroid, node2.centroid])
                    edge_colors.append([100, 100, 100, 255])  # Gray color for edges

        if edge_strips:
            rr.log("world/edges", rr.LineStrips3D(
                strips=np.array(edge_strips),
                colors=np.array(edge_colors)
            ))

        for node_name in self.nodes:
            node = self.nodes[node_name]
            if show_pct and node.pct is not None and node.visible:
                # Resample points only for visible nodes
                sampled_points = node.pct[np.random.choice(
                    node.pct.shape[0],
                    min(max_points, node.pct.shape[0]),
                    replace=False
                )]

                # Get base color from node's color (RGB 0-1 from get_color_for_id)
                base_color = node.color
                if base_color is not None and len(base_color) == 3:
                    # Convert RGB 0-1 to RGB 0-255 for rerun
                    r, g, b = base_color
                    point_color = np.array([
                        int(r * 255),
                        int(g * 255),
                        int(b * 255)
                    ])
                    # Apply color to all points
                    point_colors = np.tile(point_color, (sampled_points.shape[0], 1))
                else:
                    # Fallback to class_ids for coloring
                    point_colors = None

                rr.log(f"world/points/{node_name}", rr.Points3D(
                    positions=sampled_points,
                    colors=point_colors,
                    class_ids=np.array([node.id] * sampled_points.shape[0])
                ))

            if node.visible:
                base_color = node.color
                if base_color is not None and len(base_color) == 3:
                    # Convert RGB 0-1 to RGB 0-255 for rerun
                    r, g, b = base_color
                    centroid_color = np.array([
                        int(r * 255),
                        int(g * 255),
                        int(b * 255)
                    ])
                else:
                    # Fallback color
                    centroid_color = np.array([255, 100, 100])  # Red-orange

                rr.log(f"world/centroids/{node_name}", rr.Points3D(
                    positions=node.centroid,
                    colors=np.array([centroid_color]),
                    class_ids=np.array([node.id])
                ))

    def log_rerun_teaser(self, show_pct: bool = False, max_points: int = 3000, edge_threshold: float = 0.5,
                         timestep: int = 0, transition_start: int = 900, transition_duration: int = 200,
                         final_edge_thickness: float = 0.008):
        """
        Log the scene graph to rerun with radius-based animation transitions.

        Args:
            show_pct: Whether to show point clouds
            max_points: Maximum number of points to sample per node
            edge_threshold: Distance threshold for creating edges between nodes
            timestep: Current timestep (0-1700)
            transition_start: Timestep when transition begins (default: 900)
            transition_duration: Duration of transition in timesteps (default: 200)
            final_edge_thickness: Final thickness of edges after transition (default: 0.008)
        """
        # Calculate transition progress (0.0 to 1.0)
        transition_end = transition_start + transition_duration
        if timestep <= transition_start:
            transition_progress = 0.0
        elif timestep >= transition_end:
            transition_progress = 1.0
        else:
            transition_progress = (timestep - transition_start) / transition_duration

        # Calculate dynamic radii based on transition progress
        # Points: radius decreases from 0.010 to 0.001 during transition
        points_radius = 0.010 * (1 - transition_progress * 0.9)  # Reduces to 10% of original size

        # Centroids: radius increases from 0.001 to 0.08 during transition
        centroids_radius = 0.001 + (transition_progress * 0.079)  # Grows from tiny to full size

        # Edges: thickness increases from 0.001 to final_edge_thickness during transition
        edges_thickness = 0.001 + (transition_progress * (final_edge_thickness - 0.001))

        # Debug: print radius values for verification
        if timestep % 100 == 0:  # Print every 100 frames
            print(f"Frame {timestep}: transition_progress={transition_progress:.3f}")
            print(f"  Points radius: {points_radius:.6f}, Centroids radius: {centroids_radius:.6f}, Edges thickness: {edges_thickness:.6f}")

            # Count visible vs total nodes for debugging
            total_nodes = len(self.nodes)
            visible_nodes = sum(1 for node in self.nodes.values() if node.visible)
            print(f"  Nodes: {visible_nodes}/{total_nodes} visible")

        # Create edges between nodes that are closer than edge_threshold meters
        edge_strips = []
        edge_colors = []

        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1 = node_list[i]
                node2 = node_list[j]
                distance = np.linalg.norm(node1.centroid - node2.centroid)
                if distance < edge_threshold:
                    edge_strips.append([node1.centroid, node2.centroid])
                    edge_colors.append([100, 100, 100, 255])  # Solid gray color for edges

        if edge_strips:
            rr.log("world/edges", rr.LineStrips3D(
                strips=np.array(edge_strips),
                colors=np.array(edge_colors),
                radii=edges_thickness  # Use dynamic thickness based on transition
            ))

        for node_name in self.nodes:
            node = self.nodes[node_name]

            # During transition, render ALL nodes (even invisible ones) to apply shrinking radius
            # Outside transition, only render visible nodes as usual
            should_render_points = False
            if show_pct and node.pct is not None:
                if transition_progress > 0.0 and transition_progress < 1.0:
                    # During transition: render ALL nodes with transition radius
                    should_render_points = True
                elif node.visible:
                    # Outside transition: only render visible nodes
                    should_render_points = True

            if should_render_points:
                # Resample points only for visible nodes
                sampled_points = node.pct[np.random.choice(
                    node.pct.shape[0],
                    min(max_points, node.pct.shape[0]),
                    replace=False
                )]

                # Get base color from node's color (RGB 0-1 from get_color_for_id)
                base_color = node.color
                if base_color is not None and len(base_color) == 3:
                    # Convert RGB 0-1 to RGB 0-255 for rerun
                    r, g, b = base_color
                    point_color = np.array([
                        int(r * 255),
                        int(g * 255),
                        int(b * 255)
                    ])
                    # Apply color to all points
                    point_colors = np.tile(point_color, (sampled_points.shape[0], 1))
                else:
                    # Fallback to class_ids for coloring
                    point_colors = None

                rr.log(f"world/points/{node_name}", rr.Points3D(
                    positions=sampled_points,
                    radii=points_radius,  # Use dynamic radius based on transition
                    colors=point_colors,
                    class_ids=np.array([node.id] * sampled_points.shape[0])
                ))

            # Only render centroids once transition starts
            # During transition: render ALL centroids (even invisible ones) with growing radius
            # After transition: render ALL centroids at full size
            should_render_centroid = transition_progress > 0.0

            if should_render_centroid:
                base_color = node.color
                if base_color is not None and len(base_color) == 3:
                    # Convert RGB 0-1 to RGB 0-255 for rerun
                    r, g, b = base_color
                    centroid_color = np.array([
                        int(r * 255),
                        int(g * 255),
                        int(b * 255)
                    ])
                else:
                    # Fallback color
                    centroid_color = np.array([255, 100, 100])  # Red-orange

                rr.log(f"world/centroids/{node_name}", rr.Points3D(
                    positions=node.centroid,
                    radii=centroids_radius,  # Use dynamic radius based on transition
                    colors=np.array([centroid_color]),
                    class_ids=np.array([node.id])
                ))

    def highlight_clip_feature_similarity(self, text: str = "football", max_points: int = 5000):
        from dsg.features.clip_features import CLIPFeatures
        clip_extractor = CLIPFeatures()
        clip_features = clip_extractor.extract_text_features(text).cpu().numpy()

        cmap = plt.get_cmap("plasma")

        # collect nodes with CLIP features
        node_names = [n for n in self.nodes if self.nodes[n].clip_features is not None]

        # compute cosine similarities for all nodes
        text_feat = clip_features.squeeze()
        text_norm = np.linalg.norm(text_feat) + 1e-8
        sims = np.array([
            float(
                np.dot(self.nodes[n].clip_features, text_feat)
                / ((np.linalg.norm(self.nodes[n].clip_features) + 1e-8) * text_norm)
            )
            for n in node_names
        ])

        # robust normalization using percentiles, then gamma shaping to emphasize highs
        p_low, p_high = np.percentile(sims, [5, 95])
        denom = (p_high - p_low) if (p_high - p_low) > 1e-8 else 1.0
        norm = np.clip((sims - p_low) / denom, 0.0, 1.0)
        gamma = 0.6  # <1.0 makes top similarities stand out more
        vals = np.power(norm, gamma)

        # colorize and log
        for idx, node_name in enumerate(node_names):
            color = cmap(vals[idx])[:3]
            positions = self.nodes[node_name].pct[
                np.random.choice(
                    self.nodes[node_name].pct.shape[0],
                    min(max_points, self.nodes[node_name].pct.shape[0]),
                    replace=False,
                )
            ]
            rr.log(
                f"world/points/{node_name}",
                rr.Points3D(
                    positions=positions,
                    radii=0.005,
                    colors=np.array([color] * positions.shape[0]),
                ),
            )

    def highlight_clip_feature_similarity_progressive(self, text: str = "football", max_points: int = 5000,
                                                      num_frames: int = 60):
        """
        Create an animated visualization that progressively transitions from grey to final colors.

        Args:
            text: Text query for CLIP feature similarity
            max_points: Maximum number of points to sample per node
            num_frames: Number of animation frames
            animation_duration: Total duration of animation in seconds
        """
        from dsg.features.clip_features import CLIPFeatures
        clip_extractor = CLIPFeatures()
        clip_features = clip_extractor.extract_text_features(text).cpu().numpy()

        cmap = plt.get_cmap("plasma")

        # collect nodes with CLIP features
        node_names = [n for n in self.nodes if self.nodes[n].clip_features is not None]

        # compute cosine similarities for all nodes (same as original function)
        text_feat = clip_features.squeeze()
        text_norm = np.linalg.norm(text_feat) + 1e-8
        sims = np.array([
            float(
                np.dot(self.nodes[n].clip_features, text_feat)
                / ((np.linalg.norm(self.nodes[n].clip_features) + 1e-8) * text_norm)
            )
            for n in node_names
        ])

        # robust normalization using percentiles, then gamma shaping to emphasize highs
        p_low, p_high = np.percentile(sims, [5, 95])
        denom = (p_high - p_low) if (p_high - p_low) > 1e-8 else 1.0
        norm = np.clip((sims - p_low) / denom, 0.0, 1.0)
        gamma = 0.6  # <1.0 makes top similarities stand out more
        vals = np.power(norm, gamma)

        # Get final colors for all nodes
        final_colors = {}
        node_positions = {}

        for idx, node_name in enumerate(node_names):
            final_colors[node_name] = np.array(cmap(vals[idx])[:3])
            positions = self.nodes[node_name].pct[
                np.random.choice(
                    self.nodes[node_name].pct.shape[0],
                    min(max_points, self.nodes[node_name].pct.shape[0]),
                    replace=False,
                )
            ]
            node_positions[node_name] = positions

        # Animation parameters
        grey_color = np.array([0.5, 0.5, 0.5])  # Neutral grey

        # Create animation frames
        for frame in range(num_frames):
            # Set time for this frame
            rr.set_time_sequence("animation_frame", frame)

            # Calculate interpolation factor (0.0 = all grey, 1.0 = all final colors)
            t = frame / (num_frames - 1)  # Linear interpolation

            # Log points for this frame
            for node_name in node_names:
                # Interpolate between grey and final color
                interpolated_color = (1 - t) * grey_color + t * final_colors[node_name]

                # Apply the same color to all points in this node
                colors = np.array([interpolated_color] * node_positions[node_name].shape[0])

                rr.log(
                    f"world/points/{node_name}",
                    rr.Points3D(
                        positions=node_positions[node_name],
                        radii=0.005,
                        colors=colors,
                    ),
                )

    def log_open3d(self, vis, geometries, show_pct: bool = False, max_points: int = 5000):
        """Log scene graph to Open3D visualizer"""
        # Clear existing object geometries
        for geom in geometries['objects']:
            vis.remove_geometry(geom, False)
        geometries['objects'].clear()
        
        # Add new object geometries
        for node_name, node in self.nodes.items():
            if show_pct and node.pct is not None:
                # Sample points if too many
                if len(node.pct) > max_points:
                    indices = np.random.choice(len(node.pct), max_points, replace=False)
                    points = node.pct[indices]
                else:
                    points = node.pct
                
                # Create point cloud for object points
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(node.color / 255.0)  # Convert to [0,1] range
                
                geometries['objects'].append(pcd)
                vis.add_geometry(pcd, False)
            
            # Create centroid sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            sphere.translate(node.centroid)
            sphere.paint_uniform_color(node.color / 255.0)
            
            geometries['objects'].append(sphere)
            vis.add_geometry(sphere, False)

    def __len__(self):
        return len(self.nodes)

if __name__ == "__main__":
    graph = SceneGraph()
    graph.add_node(Node("node1", np.array([0, 0, 0])))
    graph.add_node(Node("node2", np.array([1, 0, 0])))
    graph.add_node(Node("node3", np.array([0, 1, 0])))
    graph.add_node(Node("node4", np.array([1, 1, 0])))
    rr.init("scene_graph", spawn=True)
    graph.log_rerun()