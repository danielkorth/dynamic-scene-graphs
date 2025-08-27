import numpy as np
from typing import Optional, Tuple
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

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
                 rgb_points: np.ndarray | None = None,
                 tsdf: Optional[TSDFVolume] = None,
                 use_tsdf: bool = False):
        self.name = name
        self.id = int(name.split("_")[-1])
        self.centroid = centroid
        self.color = color
        self.radius = radius
        self.label = label
        self.pct = pct
        self.clip_features = None
        self.rgb = rgb_points
        self.tsdf = tsdf
        self.use_tsdf = use_tsdf
        self.n_acc_pcs = 0
        self.max_vol = 0
        self.visible = True  # Track visibility state
    
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
    
    def add_pct(self, pct: np.ndarray, rgb_points: np.ndarray):
        if self.pct is None:
            self.pct = pct
            self.rgb = rgb_points
        else:
            self.pct = np.concatenate([self.pct, pct], axis=0)
            self.rgb = np.concatenate([self.rgb, rgb_points], axis=0)
        
    @property
    def clip_features(self):
        return self._clip_features

    @clip_features.setter
    def clip_features(self, value: np.ndarray):
        self._clip_features = value
    
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

    def _calculate_bbox_area(self, points_3d: np.ndarray) -> float:
        """
        Calculate the bounding box area of a pointcloud (for plane detection).
        
        Args:
            points_3d: Pointcloud to calculate area for (N, 3)
            
        Returns:
            Area of the bounding box in m^2, or -1 if no points
        """
        if points_3d is not None and len(points_3d) > 0:
            min_pt = np.min(points_3d, axis=0)
            max_pt = np.max(points_3d, axis=0)
            dimensions = max_pt - min_pt
            
            # Calculate area of the two largest dimensions (for a plane-like object)
            sorted_dims = np.sort(dimensions)
            thickness_ratio = sorted_dims[0] / sorted_dims[1]
            is_plane = thickness_ratio < 0.28
            return sorted_dims[1] * sorted_dims[2], is_plane  # Area of largest face
        return -1, None

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

    def _should_replace_pointcloud(self, points_3d: np.ndarray, accumulate_points: bool, 
                                 camera_intrinsics: np.ndarray = None,
                                 camera_pose: np.ndarray = None,
                                 current_mask: np.ndarray = None,
                                 depth_image: np.ndarray = None) -> str:
        """
        Determine if the pointcloud should be replaced, merged, or accumulated.
        
        Args:
            points_3d: Pointcloud data for plane detection
            accumulate_points: Whether to accumulate points
            camera_intrinsics: Camera intrinsic matrix (3, 3) for motion detection
            camera_pose: Camera pose matrix (4, 4) for motion detection
            current_mask: Current object mask (H, W) for motion detection
            depth_image: Depth image (H, W) for occlusion checking in motion detection
            
        Returns:
            "replace", "merge", "accumulate", or "ignore"
        """
        # Replace if not accumulating or if no existing pointcloud
        if (not accumulate_points or 
                self.pct is None or 
                len(self.pct) == 0 or
                self.n_acc_pcs >= 1000):
            return "replace"

        VOLUME_THRESHOLD = 0.005  # adjust as needed
        AREA_THRESHOLD = 0.01  # 5cm² threshold for plane area
        OCC_THRESHOLD = 0.3
        SIZE_WRT_PREV = 0.2
        
        # Check if this might be a plane
        bbox_volume = self._calculate_bbox_volume(points_3d)
        bbox_area, is_plane = self._calculate_bbox_area(points_3d)
        
        # For planes, use area instead of volume
        if is_plane and bbox_area > AREA_THRESHOLD:
            print(f"Plane detected: volume={bbox_volume:.6f}, area={bbox_area:.6f}, threshold={AREA_THRESHOLD}")
            occ_percentage = calculate_occlusion_percentage(self.pct, current_mask, depth_image, camera_intrinsics, camera_pose, depth_tolerance=0.01, visualize=False)
            print(f"Occ percentage: {occ_percentage}")
            if occ_percentage > OCC_THRESHOLD:
                return "ignore"
            else:
                return "accumulate" 
        elif is_plane and bbox_area < SIZE_WRT_PREV*self._calculate_bbox_area(self.pct)[0]:
            print(f"Plane detected but area small wrt previous -> ignoring")
            return "ignore"
        
        # If volume is too large, replace instead of accumulate
        if bbox_volume > self.max_vol:
                self.max_vol = bbox_volume
        if self.max_vol > VOLUME_THRESHOLD and not self.is_object_moving(camera_intrinsics, camera_pose, current_mask, depth_image, depth_tolerance=0.01, visualize=False):
            return "accumulate"

        occ_percentage = calculate_occlusion_percentage(self.pct, current_mask, depth_image, camera_intrinsics, camera_pose, depth_tolerance=0.01, visualize=False)
        if bbox_volume < SIZE_WRT_PREV*self.max_vol or occ_percentage > OCC_THRESHOLD:
            print(f"Volume small wrt previous -> ignoring")
            print(f"Occ percentage: {occ_percentage}")
            return "ignore"
        
        # Check if object is moving to decide between merge and accumulate
        if (camera_intrinsics is not None and 
            camera_pose is not None and 
            current_mask is not None and
            depth_image is not None):
            
            # is_moving = self.is_object_moving(camera_intrinsics, camera_pose, current_mask, depth_image)
            is_moving = True
            # print(f"Object is moving: {is_moving}")
            if is_moving:
                return "merge"  # Merge if object is moving
            else:
                return "accumulate"  # Accumulate if object is stationary
        
        # Default to merge if motion detection parameters are not provided
        return "merge"

    def _replace_pointcloud(self, points_3d: np.ndarray, rgb_points: np.ndarray):
        """
        Replace the current pointcloud with new points.
        
        Args:
            points_3d: New pointcloud to set (N, 3)
        """
        self.pct = points_3d
        self.rgb = rgb_points
        self.n_acc_pcs = 1

    def _register_and_merge_pointclouds(self, points_3d: np.ndarray, rgb_points: np.ndarray):
        """
        Register and merge new pointcloud with existing one using global registration and ICP.
        
        Args:
            points_3d: New pointcloud to merge (N, 3)
        """
        # Convert numpy arrays to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(self.pct)
        source_pcd.colors = o3d.utility.Vector3dVector(self.rgb)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d)
        target_pcd.colors = o3d.utility.Vector3dVector(rgb_points)

        # Decide whether to run global registration based on centroid distance vs. a
        # dynamic threshold that depends on the volumes of the two pointclouds
        source_centroid = np.mean(self.pct, axis=0) if self.pct is not None and len(self.pct) > 0 else None
        target_centroid = np.mean(points_3d, axis=0) if points_3d is not None and len(points_3d) > 0 else None

        # Default to running global registration if we cannot compute centroids
        run_global_registration = True
        if source_centroid is not None and target_centroid is not None:
            centroid_distance = float(np.linalg.norm(source_centroid - target_centroid))

            # Compute a characteristic size from bounding box volumes (fallback to bbox diagonal)
            source_volume = self._calculate_bbox_volume(self.pct)
            target_volume = self._calculate_bbox_volume(points_3d)

            def characteristic_size(pts: np.ndarray, volume: float) -> float:
                if volume is not None and volume > 0:
                    return float(np.cbrt(volume))
                if pts is not None and len(pts) > 0:
                    min_pt = np.min(pts, axis=0)
                    max_pt = np.max(pts, axis=0)
                    return float(np.linalg.norm(max_pt - min_pt))
                return 0.0

            size_scale = max(characteristic_size(self.pct, source_volume),
                              characteristic_size(points_3d, target_volume))
            # Threshold is a fraction of the characteristic size, clamped to sensible bounds (meters)
            centroid_threshold = float(np.clip(0.2 * size_scale, 0.005, 0.05))

            # If centroids are already close relative to object size, skip global registration
            run_global_registration = centroid_distance > centroid_threshold

        initial_transformation = np.eye(4)
        if run_global_registration:
            # Prepare point clouds for global registration
            source_down, source_fpfh = prep_pc_for_global_registration(source_pcd, 0.005)
            target_down, target_fpfh = prep_pc_for_global_registration(target_pcd, 0.005)

            # Perform global registration
            global_registration_result = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, 0.05
            )
            initial_transformation = global_registration_result.transformation

        # Estimate normals for both point clouds (required for ICP)
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Perform ICP with global registration result as initial transformation
        distance_threshold = 0.01  # 1cm threshold

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
        self.rgb = np.vstack([rgb_points, self.rgb])
        self.n_acc_pcs += 1
        

    def integrate_pointcloud(self, points_3d: np.ndarray, rgb_points: np.ndarray, accumulate_points: bool = False, visualize: bool = False,
                           camera_intrinsics: np.ndarray = None, camera_pose: np.ndarray = None, 
                           current_mask: np.ndarray = None, depth_image: np.ndarray = None):
        """
        Integrate new pointcloud using global registration followed by ICP.
        Also visualize the input pointcloud and its bounding box volume.

        Args:
            points_3d: New pointcloud to integrate (N, 3)
            accumulate_points: If True, perform registration and merge. If False, replace existing pointcloud.
            visualize: If True, visualize the input pointcloud and its bounding box.
            camera_intrinsics: Camera intrinsic matrix (3, 3) for motion detection
            camera_pose: Camera pose matrix (4, 4) for motion detection
            current_mask: Current object mask (H, W) for motion detection
            depth_image: Depth image (H, W) for occlusion checking in motion detection
        """
    
        # Check if we should replace the pointcloud instead of accumulating
        aggregation = self._should_replace_pointcloud(points_3d, accumulate_points, 
                                                    camera_intrinsics, camera_pose, current_mask, depth_image)

        # Visualization of the input pointcloud and its bounding box
        if visualize:
            # Calculate bounding box volume
            bbox_volume = self._calculate_bbox_volume(points_3d)
            self._visualize_pointcloud_and_bbox(points_3d, bbox_volume)

        # aggregation = "accumulate"

        if aggregation == "replace":
            self._replace_pointcloud(points_3d, rgb_points)
        elif aggregation == "merge":
            self._register_and_merge_pointclouds(points_3d, rgb_points)
        elif aggregation == "accumulate":
            self.add_pct(points_3d, rgb_points)
        elif aggregation == "ignore":
            return
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        # Voxel downsample before computing centroid
        o3d_pct = o3d.geometry.PointCloud()
        o3d_pct.points = o3d.utility.Vector3dVector(self.pct)
        o3d_pct.colors = o3d.utility.Vector3dVector(self.rgb)   
        
        # Compute bounding box volume to determine voxel size
        if self.pct is not None and len(self.pct) > 0:
            min_pt = np.min(self.pct, axis=0)
            max_pt = np.max(self.pct, axis=0)
            bbox_diag = np.linalg.norm(max_pt - min_pt)
            # Set voxel size as a fraction of the bounding box diagonal (e.g., 1/100th)
            voxel_size = np.clip(bbox_diag / 100.0, 0.0005, 0.1)
            # voxel_size = 0.01
            # voxel_size = max(bbox_diag / 100.0, 0.005)
        else:
            voxel_size = 0.01  # fallback default
        
        # Voxel downsample
        pcd_down = o3d_pct.voxel_down_sample(voxel_size)
        # pcd_down = o3d_pct

        # Statistical outlier removal
        std_ratio = 4.0 if aggregation == "accumulate" else 2.0
        if len(pcd_down.points) > 0 and aggregation != "ignore":
            # nb_neighbors and std_ratio can be tuned as needed
            cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
            pcd_filtered = pcd_down.select_by_index(ind)
            self.pct = np.asarray(pcd_filtered.points)
            self.rgb = np.asarray(pcd_filtered.colors)
        else:
            self.pct = np.asarray(pcd_down.points)
            self.rgb = np.asarray(pcd_down.colors)
        
        self.centroid = np.mean(self.pct, axis=0)

    def is_object_moving(self, 
                        camera_intrinsics: np.ndarray,
                        camera_pose: np.ndarray,
                        current_mask: np.ndarray,
                        depth_image: np.ndarray,
                        outside_points_threshold: float = 0.5,
                        depth_tolerance: float = 0.01,
                        visualize: bool = False) -> bool:
        """
        Detect if an object is moving by counting projected points that fall outside the current mask.
        Includes occlusion checking using depth information.
        
        Args:
            camera_intrinsics: Camera intrinsic matrix (3, 3)
            camera_pose: Camera pose matrix (4, 4) - world to camera transform
            current_mask: Current object mask (H, W) - boolean array
            depth_image: Depth image (H, W) - depth values in meters
            outside_points_threshold: Percentage of points outside mask to consider object moving (default: 0.5)
            depth_tolerance: Tolerance for depth comparison in meters (default: 0.01)
            visualize: Whether to show visualization of mask and projected points (default: False)
            
        Returns:
            True if object is moving (more than threshold percentage of points outside mask), False otherwise
        """
        if self.pct is None or len(self.pct) == 0:
            return False

        # Early termination: if threshold is 0, any outside points mean moving
        if outside_points_threshold == 0:
            return True

        # Dilate the current_mask before proceeding
        dilate_kernel = np.ones((7, 7), np.uint8)
        current_mask = cv2.dilate(current_mask.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)
            
        # Transform pointcloud from world coordinates to camera coordinates
        pose_inv = np.linalg.inv(camera_pose)
        points_camera = (pose_inv[:3, :3] @ self.pct.T).T + pose_inv[:3, 3]
        
        # Early termination: check if any points are in front of camera
        valid_mask = points_camera[:, 2] > 0
        if not np.any(valid_mask):
            return False
            
        # Get camera parameters once
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        height, width = current_mask.shape
        
        # Vectorized projection for valid points only
        valid_points = points_camera[valid_mask]
        points_2d = np.zeros((len(points_camera), 2))
        points_2d[valid_mask, 0] = fx * valid_points[:, 0] / valid_points[:, 2] + cx
        points_2d[valid_mask, 1] = fy * valid_points[:, 1] / valid_points[:, 2] + cy
        
        # Filter points that fall outside the camera frame (camera frustum)
        in_frame_mask = (
            (points_2d[:, 0] >= 0) & 
            (points_2d[:, 0] < width) & 
            (points_2d[:, 1] >= 0) & 
            (points_2d[:, 1] < height) &
            valid_mask
        )
        
        if not np.any(in_frame_mask):
            return False
            
        # Get all valid points at once
        points_2d_int = points_2d[in_frame_mask].astype(int)
        points_camera_in_frame = points_camera[in_frame_mask]
        
        # Vectorized depth visibility checking
        y_coords = points_2d_int[:, 1]
        x_coords = points_2d_int[:, 0]
        
        # Get depth values from depth image
        depth_image = depth_image / 1000.0 # convert to meters
        depth_at_pixels = depth_image[y_coords, x_coords]
        point_depths = points_camera_in_frame[:, 2]
        
        # Vectorized visibility check - optimized boolean operations
        valid_depth_mask = depth_at_pixels > 0
        visible_mask = np.ones(len(points_2d_int), dtype=bool)
        visible_mask[valid_depth_mask] = point_depths[valid_depth_mask] <= (depth_at_pixels[valid_depth_mask] + depth_tolerance)
        
        # Early termination: check if any points are visible
        if not np.any(visible_mask):
            return False
        
        # Get visible points - avoid creating new array if not needed for visualization
        if visualize:
            visible_points_2d_int = points_2d_int[visible_mask]
            y_coords_visible = visible_points_2d_int[:, 1]
            x_coords_visible = visible_points_2d_int[:, 0]
        else:
            # Use boolean indexing directly for better performance
            y_coords_visible = y_coords[visible_mask]
            x_coords_visible = x_coords[visible_mask]
        
        # Get mask values for all visible points at once
        mask_values = current_mask[y_coords_visible, x_coords_visible]
        
        # Count points outside mask
        outside_mask_count = np.sum(~mask_values)
        total_points = len(mask_values)
        
        if total_points == 0:
            return False
            
        outside_percentage = outside_mask_count / total_points
        
        # Early termination: if we have enough outside points, we can stop
        if outside_percentage > outside_points_threshold:
            moving = True
        else:
            moving = False
        
        # Visualization (only if needed and moving)
        if visualize and moving:
            # Prepare visualization data
            visible_points_2d_int = points_2d_int[visible_mask]
            inside_points = visible_points_2d_int[mask_values].tolist()
            outside_points = visible_points_2d_int[~mask_values].tolist()
            visualize_mask_and_points(current_mask, inside_points, outside_points, 
                                          outside_percentage*100, outside_points_threshold*100)
                
        # Object is considered moving if more than threshold percentage of points are outside the mask
        if moving:
            print(f"Object {self.name} is moving with {outside_percentage*100:.1f}% outside points ({outside_mask_count}/{total_points})")
        return moving
    

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value

class Edge:
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target

def calculate_occlusion_percentage(points_3d: np.ndarray,
                                 mask: np.ndarray,
                                 depth_image: np.ndarray,
                                 camera_intrinsics: np.ndarray,
                                 camera_pose: np.ndarray,
                                 depth_tolerance: float = 0.01,
                                 visualize: bool = False) -> float:
    """
    Calculate the percentage of 3D points that are occluded inside a given mask.
    
    This function projects 3D points to 2D image coordinates, checks if they fall within
    the mask, and then determines if they are occluded by comparing their depth with
    the depth image values.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        mask: Object mask (H, W) - boolean array where True indicates inside mask
        depth_image: Depth image (H, W) - depth values in millimeters
        camera_intrinsics: Camera intrinsic matrix (3, 3)
        camera_pose: Camera pose matrix (4, 4) - world to camera transform
        depth_tolerance: Tolerance for depth comparison in meters (default: 0.05)
        
    Returns:
        float: Percentage of points inside the mask that are occluded (0.0 to 1.0)
               Returns 0.0 if no points fall inside the mask
    """
    if len(points_3d) == 0:
        return 0.0
    
    # Transform pointcloud from world coordinates to camera coordinates
    pose_inv = np.linalg.inv(camera_pose)
    points_camera = (pose_inv[:3, :3] @ points_3d.T).T + pose_inv[:3, 3]
    
    # Check if any points are in front of camera
    valid_mask = points_camera[:, 2] > 0
    if not np.any(valid_mask):
        return 0.0
        
    # Get camera parameters
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    height, width = mask.shape
    
    # Vectorized projection for valid points only
    valid_points = points_camera[valid_mask]
    points_2d = np.zeros((len(points_camera), 2))
    points_2d[valid_mask, 0] = fx * valid_points[:, 0] / valid_points[:, 2] + cx
    points_2d[valid_mask, 1] = fy * valid_points[:, 1] / valid_points[:, 2] + cy
    
    # Filter points that fall within the camera frame
    in_frame_mask = (
        (points_2d[:, 0] >= 0) & 
        (points_2d[:, 0] < width) & 
        (points_2d[:, 1] >= 0) & 
        (points_2d[:, 1] < height) &
        valid_mask
    )
    
    if not np.any(in_frame_mask):
        return 0.0
        
    # Get all valid points that are in frame
    points_2d_int = points_2d[in_frame_mask].astype(int)
    points_camera_in_frame = points_camera[in_frame_mask]
    
    # Get 2D coordinates
    y_coords = points_2d_int[:, 1]
    x_coords = points_2d_int[:, 0]
    
    # Check which points fall inside the mask
    mask_values = mask[y_coords, x_coords]
    inside_mask_mask = mask_values
    
    # If no points fall inside the mask, return 0
    if not np.any(inside_mask_mask):
        return 0.0
    
    # Convert depth image to meters and get depth values at these pixels
    depth_image_meters = depth_image / 1000.0  # convert from mm to meters
    depth_at_pixels = depth_image_meters[y_coords, x_coords]
    point_depths = points_camera_in_frame[:, 2]
    
    # Check which points have valid depth values in the depth image
    valid_depth_mask = depth_at_pixels > 0
    
    # If no valid depth values, assume all points are visible
    if not np.any(valid_depth_mask):
        return 0.0
    
    # Check occlusion: a point is occluded if its depth is greater than the depth image value
    # (meaning there's something closer to the camera at that pixel)
    # Add tolerance to account for noise
    occluded_mask = np.zeros(len(points_2d_int), dtype=bool)
    occluded_mask[valid_depth_mask] = (point_depths[valid_depth_mask] > (depth_at_pixels[valid_depth_mask] + depth_tolerance)) & (~inside_mask_mask[valid_depth_mask])
    
    # Calculate percentage of occluded points among points inside mask
    total_points_inside_mask = len(points_2d_int)
    occluded_points = np.sum(occluded_mask)
    
    occlusion_percentage = occluded_points / total_points_inside_mask

    # Visualization
    if visualize:
        # Points inside mask and not occluded (green), inside mask and occluded (red)
        inside_not_occluded = points_2d_int[~occluded_mask]
        inside_occluded = points_2d_int[occluded_mask]
        # Use the mask as current_mask, inside_points as not occluded, outside_points as occluded
        visualize_mask_and_points(
            mask,
            inside_not_occluded.tolist(),
            inside_occluded.tolist(),
            occlusion_percentage * 100,
            0.0  # No threshold for occlusion, just show percentage
        )

    return occlusion_percentage

def visualize_mask_and_points(current_mask: np.ndarray,
                                 inside_points: list,
                                 outside_points: list,
                                 outside_percentage: float,
                                 outside_points_threshold_percent: float) -> None:
        """
        Visualize the binary mask with projected points overlaid.
        
        Args:
            current_mask: Current object mask (H, W) - boolean array
            inside_points: List of [x, y] coordinates of points inside the mask
            outside_points: List of [x, y] coordinates of points outside the mask
            outside_percentage: Percentage of points outside the mask
            outside_points_threshold_percent: Percentage threshold for considering object moving
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Display the binary mask
        plt.imshow(current_mask, cmap='gray', alpha=0.7)
        
        # Plot points inside the mask in green
        if inside_points:
            inside_points = np.array(inside_points)
            plt.scatter(inside_points[:, 0], inside_points[:, 1], 
                       c='green', s=20, alpha=0.8, label=f'Inside mask ({len(inside_points)} points)')
        
        # Plot points outside the mask in red
        if outside_points:
            outside_points = np.array(outside_points)
            plt.scatter(outside_points[:, 0], outside_points[:, 1], 
                       c='red', s=20, alpha=0.8, label=f'Outside mask ({len(outside_points)} points)')
        
        # Add title and labels
        is_moving = outside_percentage > outside_points_threshold_percent
        status = "MOVING" if is_moving else "STATIONARY"
        plt.title(f'Object Motion Detection - {status}\n'
                 f'Outside points: {outside_percentage:.1f}%/{outside_points_threshold_percent:.1f}% threshold')
        
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text annotation with statistics
        total_points = len(inside_points) + len(outside_points)
        if total_points > 0:
            plt.text(0.02, 0.98, f'Total projected points: {total_points}\n'
                     f'Outside percentage: {outside_percentage:.1f}%\n'
                     f'Threshold: {outside_points_threshold_percent:.1f}%', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()