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

class Edge:
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target
