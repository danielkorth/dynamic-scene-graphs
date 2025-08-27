from dsg.scenegraph.node import Node, Edge
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from utils.tools import get_color_for_id
from scipy.spatial.transform import Rotation
from utils.cv2_utils import unproject_image

def process_frame_with_representation(rgb, depth, tvec, rvec, obj_points, K, graph, cfg, use_tsdf=False):
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
        points_3d, pixel_coords = unproject_image(depth, K, -rvec, tvec, mask=obj_point['mask'], dist=None, mask_erode_kernel=5)
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
                                            current_mask=obj_point['mask'], depth_image=depth)
            

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
                    edge_colors.append([255, 255, 255, 255])  # White color for edges

        if edge_strips:
            rr.log("world/edges", rr.LineStrips3D(
                strips=np.array(edge_strips),
                colors=np.array(edge_colors)
            ))

        for node_name in self.nodes:
            if show_pct:
                rr.log(f"world/points/{node_name}", rr.Points3D(
                    # sample max_points points
                    positions=self.nodes[node_name].pct[np.random.choice(self.nodes[node_name].pct.shape[0], min(max_points, self.nodes[node_name].pct.shape[0]), replace=False)], 
                    radii=0.005,
                    class_ids=np.array([self.nodes[node_name].id] * self.nodes[node_name].pct.shape[0]))
                )
            rr.log(f"world/centroids/{node_name}", rr.Points3D(
                positions=self.nodes[node_name].centroid, 
                radii=0.03,
                class_ids=np.array([self.nodes[node_name].id] * self.nodes[node_name].pct.shape[0]))
            )

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