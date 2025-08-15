from dsg.scenegraph.node import Node, Edge
import rerun as rr
import numpy as np
import open3d as o3d
from dsg.utils.tools import get_color_for_id
from scipy.spatial.transform import Rotation
from dsg.utils.cv2_utils import unproject_image
import cv2

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
    
    def log_rerun(self, show_pct: bool = False, max_points: int = 5000):
        # edges = rr.LineStrips3D(strips=np.array([[edge.source.centroid, edge.target.centroid] for edge in self.edges]))
        # rr.log("world/edges", edges)
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