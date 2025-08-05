import numpy as np
from typing import Optional, Tuple
import open3d as o3d


class TSDFVolume:
    """TSDF (Truncated Signed Distance Function) volume using Open3D implementation."""
    
    def __init__(self, 
                 voxel_size: float = 0.01,
                 volume_size: float = 1.0,
                 truncation_distance: float = 0.05):
        """
        Initialize TSDF volume using Open3D.
        
        Args:
            voxel_size: Size of each voxel in meters
            volume_size: Size of the volume cube in meters
            truncation_distance: Truncation distance for the TSDF
        """
        self.voxel_size = voxel_size
        self.volume_size = volume_size
        self.truncation_distance = truncation_distance
        
        # Calculate grid dimensions
        self.grid_size = int(volume_size / voxel_size)
        
        # Initialize Open3D TSDF volume
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=truncation_distance,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Volume origin (center of the volume)
        self.origin = np.array([-volume_size/2, -volume_size/2, -volume_size/2])
        
        # Track if volume has been initialized
        self._initialized = False
    
    def integrate_frame(self, 
                       depth_image: np.ndarray,
                       rgb_image: np.ndarray,
                       mask: np.ndarray,
                       camera_intrinsics: np.ndarray,
                       camera_pose: np.ndarray) -> None:
        """
        Integrate a new RGB-D frame into the TSDF volume using Open3D.
        
        Args:
            depth_image: Depth image (H, W) in meters
            rgb_image: RGB image (H, W, 3) uint8
            mask: Object mask (H, W) - boolean array
            camera_intrinsics: Camera intrinsic matrix (3, 3)
            camera_pose: Camera pose matrix (4, 4) - world to camera transform
        """
        # Apply mask to depth and RGB images
        masked_depth = depth_image.copy()
        masked_depth[~mask] = 0
        
        masked_rgb = rgb_image.copy()
        masked_rgb[~mask] = 0
        
        # Convert to Open3D format
        depth_o3d = o3d.geometry.Image(masked_depth.astype(np.float32))
        color_o3d = o3d.geometry.Image(masked_rgb)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,  # Depth is already in meters
            depth_trunc=10.0,  # Maximum depth to consider
            convert_rgb_to_intensity=False
        )
        
        # Create intrinsic matrix for Open3D
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=np.asarray(depth_image).shape[1], 
                                                        height=np.asarray(depth_image).shape[0], 
                                                        intrinsic_matrix=camera_intrinsics)
        
        # Create camera pose for Open3D (convert from world-to-camera to camera-to-world)
        camera_to_world = np.linalg.inv(camera_pose)
        
        # Integrate into TSDF volume
        self.tsdf_volume.integrate(rgbd, intrinsic, camera_to_world)
        self._initialized = True
    
    def extract_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract point cloud from TSDF volume using Open3D.

        Returns:
            Tuple of (points, colors) where points are (N, 3) and colors are (N, 3)
        """
        if not self._initialized:
            return np.array([]), np.array([])
        
        # Extract mesh from TSDF volume
        mesh = self.tsdf_volume.extract_triangle_mesh()
        
        if len(mesh.vertices) == 0:
            return np.array([]), np.array([])
        
        # Convert mesh vertices to numpy array
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors) * 255  # Convert from [0,1] to [0,255]
        colors = colors.astype(np.uint8)
        
        return points, colors
    
    def extract_voxel_grid(self) -> Optional[o3d.geometry.VoxelGrid]:
        """
        Extract voxel grid from TSDF volume.
        
        Returns:
            Open3D VoxelGrid object or None if not initialized
        """
        if not self._initialized:
            return None
        
        return self.tsdf_volume.extract_voxel_grid()
    
    def get_volume_size(self) -> Tuple[int, int, int]:
        """Get the size of the TSDF volume grid."""
        return (self.grid_size, self.grid_size, self.grid_size)

def prep_pc_for_global_registration(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


class Node:
    def __init__(self, 
                 name: str, 
                 centroid: np.ndarray, 
                 color: np.ndarray = np.array([0, 0, 0]),
                 radius: float = 0.003,
                 label: str = "",
                 pct: np.ndarray | None = None,
                 tsdf: Optional[TSDFVolume] = None,
                 use_tsdf: bool = False):
        self.name = name
        self.id = int(name.split("_")[-1])
        self.centroid = centroid
        self.color = color
        self.radius = radius
        self.label = label
        self.pct = pct
        self.tsdf = tsdf
        self.use_tsdf = use_tsdf
        self.n_acc_pcs = 0
        self.max_vol = 0
    
    @property
    def centroid(self):
        return self._centroid
    
    @centroid.setter
    def centroid(self, value: np.ndarray):
        self._centroid = value
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value: np.ndarray):
        self._color = value
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value: float):
        self._radius = value
    
    @property
    def pct(self):
        return self._pct
    
    @pct.setter
    def pct(self, value: np.ndarray):
        self._pct = value
    
    def add_pct(self, pct: np.ndarray):
        if self.pct is None:
            self.pct = pct
        else:
            self.pct = np.concatenate([self.pct, pct], axis=0)
    
    def integrate_frame_to_tsdf(self, 
                               depth_image: np.ndarray,
                               rgb_image: np.ndarray,
                               mask: np.ndarray,
                               camera_intrinsics: np.ndarray,
                               camera_pose: np.ndarray) -> None:
        """Integrate a new frame into the TSDF volume using Open3D."""
        if self.tsdf is None:
            # Initialize TSDF volume with appropriate size based on object
            # Use a larger volume size to accommodate object movement
            volume_size = 1.0  # 1 meter cube
            self.tsdf = TSDFVolume(voxel_size=0.01, volume_size=volume_size)
        
        # Ensure depth is in meters (convert from mm if necessary)
        if depth_image.max() > 100:  # Likely in mm
            depth_image = depth_image / 1000.0
        
        self.tsdf.integrate_frame(depth_image, rgb_image, mask, camera_intrinsics, camera_pose)
        
        # Update point cloud from TSDF if using TSDF representation
        if self.use_tsdf:
            points, colors = self.tsdf.extract_point_cloud()
            if len(points) > 0:
                self.pct = points

    def _calculate_bbox_volume(self, points_3d: np.ndarray) -> float:
        """
        Calculate the bounding box volume of a pointcloud.
        
        Args:
            points_3d: Pointcloud to calculate volume for (N, 3)
            
        Returns:
            Volume of the bounding box in m^3, or -1 if no points
        """
        if points_3d is not None and len(points_3d) > 0:
            min_pt = np.min(points_3d, axis=0)
            max_pt = np.max(points_3d, axis=0)
            return np.prod(max_pt - min_pt)
        return -1

    def _visualize_pointcloud_and_bbox(self, points_3d: np.ndarray, bbox_volume: float):
        """
        Visualize the input pointcloud and its bounding box.
        
        Args:
            points_3d: Pointcloud to visualize (N, 3)
            bbox_volume: Volume of the bounding box
        """
        if points_3d is not None and len(points_3d) > 0:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # Create bounding box
            min_pt = np.min(points_3d, axis=0)
            max_pt = np.max(points_3d, axis=0)
            aabb = o3d.geometry.AxisAlignedBoundingBox(min_pt, max_pt)
            aabb.color = (1, 0, 0)
            
            print(f"Input pointcloud volume: {bbox_volume:.4f} m^3")
            o3d.visualization.draw_geometries([pcd, aabb], window_name="Input Pointcloud and Bounding Box")

    def _should_replace_pointcloud(self, bbox_volume: float, accumulate_points: bool, 
                                 camera_intrinsics: np.ndarray = None,
                                 camera_pose: np.ndarray = None,
                                 current_mask: np.ndarray = None) -> bool:
        """
        Determine if the pointcloud should be replaced instead of accumulated.
        
        Args:
            bbox_volume: Volume of the new pointcloud
            accumulate_points: Whether to accumulate points
            camera_intrinsics: Camera intrinsic matrix (3, 3) for motion detection
            camera_pose: Camera pose matrix (4, 4) for motion detection
            current_mask: Current object mask (H, W) for motion detection
            
        Returns:
            True if pointcloud should be replaced, False otherwise
        """
        VOLUME_THRESHOLD = 0.05  # adjust as needed
        
        # If volume is too large, replace instead of accumulate
        if bbox_volume > VOLUME_THRESHOLD or self.max_vol > VOLUME_THRESHOLD:
            if bbox_volume > self.max_vol:
                return True
            return False
        
        # Check if object is moving (if motion detection parameters are provided)
        if (camera_intrinsics is not None and 
            camera_pose is not None and 
            current_mask is not None):
            if self.is_object_moving(camera_intrinsics, camera_pose, current_mask):
                return True
        
        # Replace if not accumulating or if we've accumulated too many
        if (not accumulate_points or 
                self.pct is None or 
                len(self.pct) == 0 or
                self.n_acc_pcs > 10):
            return True
            
        return False

    def _replace_pointcloud(self, points_3d: np.ndarray):
        """
        Replace the current pointcloud with new points.
        
        Args:
            points_3d: New pointcloud to set (N, 3)
        """
        self.pct = points_3d
        self.n_acc_pcs = 1
        if self.pct is not None and len(self.pct) > 0:
            self.centroid = np.mean(self.pct, axis=0)

    def _register_and_merge_pointclouds(self, points_3d: np.ndarray):
        """
        Register and merge new pointcloud with existing one using global registration and ICP.
        
        Args:
            points_3d: New pointcloud to merge (N, 3)
        """
        # Convert numpy arrays to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(self.pct)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d)

        # Prepare point clouds for global registration
        source_down, source_fpfh = prep_pc_for_global_registration(source_pcd, 0.005)
        target_down, target_fpfh = prep_pc_for_global_registration(target_pcd, 0.005)

        # Perform global registration
        global_registration_result = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, 0.05
        )

        # Estimate normals for both point clouds (required for ICP)
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Perform ICP with global registration result as initial transformation
        distance_threshold = 0.01  # 1cm threshold
        initial_transformation = global_registration_result.transformation

        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 
            distance_threshold, 
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        # Apply the transformation to the existing points
        transformation_matrix = icp_result.transformation
        transformed_points = (transformation_matrix[:3, :3] @ np.asarray(source_pcd.points).T).T + transformation_matrix[:3, 3]

        # Stack the new points with the transformed old points
        self.pct = np.vstack([points_3d, transformed_points])
        self.n_acc_pcs += 1
        
        # Update centroid for the new combined pointcloud
        if len(self.pct) > 0:
            self.centroid = np.mean(self.pct, axis=0)

    def integrate_pointcloud(self, points_3d: np.ndarray, accumulate_points: bool = False, visualize: bool = False):
        """
        Integrate new pointcloud using global registration followed by ICP.
        Also visualize the input pointcloud and its bounding box volume.

        Args:
            points_3d: New pointcloud to integrate (N, 3)
            accumulate_points: If True, perform registration and merge. If False, replace existing pointcloud.
            visualize: If True, visualize the input pointcloud and its bounding box.
        """
        # Calculate bounding box volume
        bbox_volume = self._calculate_bbox_volume(points_3d)

        # Visualization of the input pointcloud and its bounding box
        if visualize:
            self._visualize_pointcloud_and_bbox(points_3d, bbox_volume)

        # Check if we should replace the pointcloud instead of accumulating
        if self._should_replace_pointcloud(bbox_volume, accumulate_points):
            self._replace_pointcloud(points_3d)
            return

        # Register and merge pointclouds
        self._register_and_merge_pointclouds(points_3d)

    def is_object_moving(self, 
                        camera_intrinsics: np.ndarray,
                        camera_pose: np.ndarray,
                        current_mask: np.ndarray,
                        iou_threshold: float = 0.3) -> bool:
        """
        Detect if an object is moving by projecting pointcloud to camera plane and computing IOU with current mask.
        
        Args:
            camera_intrinsics: Camera intrinsic matrix (3, 3)
            camera_pose: Camera pose matrix (4, 4) - world to camera transform
            current_mask: Current object mask (H, W) - boolean array
            iou_threshold: Threshold for IOU below which object is considered moving (default: 0.3)
            
        Returns:
            True if object is moving (low IOU), False otherwise
        """
        if self.pct is None or len(self.pct) == 0:
            return False
            
        # Transform pointcloud from world coordinates to camera coordinates
        camera_to_world = np.linalg.inv(camera_pose)
        points_camera = (camera_pose[:3, :3] @ self.pct.T).T + camera_pose[:3, 3]
        
        # Project 3D points to 2D image coordinates
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Perspective projection
        points_2d = np.zeros((len(points_camera), 2))
        valid_mask = points_camera[:, 2] > 0  # Only points in front of camera
        
        if not np.any(valid_mask):
            return False
            
        points_2d[valid_mask, 0] = fx * points_camera[valid_mask, 0] / points_camera[valid_mask, 2] + cx
        points_2d[valid_mask, 1] = fy * points_camera[valid_mask, 1] / points_camera[valid_mask, 2] + cy
        
        # Filter points that fall outside the camera frame
        height, width = current_mask.shape
        in_frame_mask = (
            (points_2d[:, 0] >= 0) & 
            (points_2d[:, 0] < width) & 
            (points_2d[:, 1] >= 0) & 
            (points_2d[:, 1] < height) &
            valid_mask
        )
        
        if not np.any(in_frame_mask):
            return False
            
        # Convert to integer coordinates for indexing
        points_2d_int = points_2d[in_frame_mask].astype(int)
        
        # Create a mask from the projected points
        projected_mask = np.zeros_like(current_mask, dtype=bool)
        projected_mask[points_2d_int[:, 1], points_2d_int[:, 0]] = True
        
        # Compute intersection and union
        intersection = np.logical_and(projected_mask, current_mask)
        union = np.logical_or(projected_mask, current_mask)
        
        # Calculate IOU
        intersection_count = np.sum(intersection)
        union_count = np.sum(union)
        
        if union_count == 0:
            return False
            
        iou = intersection_count / union_count
        
        # Object is considered moving if IOU is below threshold
        return iou < iou_threshold

class Edge:
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target
