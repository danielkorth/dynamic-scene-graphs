import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation
import cv2

from util.tools import center_crop

def create_camera_mesh(scale=0.1, color = [0.9, 0.1, 0.1]):
    """
    Create a rectangular-based camera mesh (frustum) to represent the camera's pose.
    The base is a rectangle, with horizontal sides longer than vertical sides.

    Parameters
    ----------
    scale : float, default=0.1
        The size of the camera frustum.

    Returns
    -------
    o3d.geometry.LineSet
        A wireframe mesh representing the camera frustum.
    """
    import numpy as np
    import open3d as o3d

    # Define the aspect ratio of the base: width > height
    width = 2.0   # horizontal (x direction)
    height = 1.0  # vertical (y direction)
    depth = 2.0   # distance from camera center to base

    # Vertices: [camera center, top-right, top-left, bottom-left, bottom-right]
    vertices = np.array([
        [0, 0, 0],  # Camera center
        [ width/2,  height/2, depth],  # Top-right
        [-width/2,  height/2, depth],  # Top-left
        [-width/2, -height/2, depth],  # Bottom-left
        [ width/2, -height/2, depth],  # Bottom-right
    ]) * scale

    # Define the edges (lines) between vertices
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from camera center to base corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Rectangle base edges
    ]

    # Create a LineSet for the camera
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set the color of the lines
    colors = [color for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def rt_to_mat(rot, trans):
    # Convert rotation vector to rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans

    return T

def mat_to_rt(T):
    """
    Inverse of rt_to_mat: extracts rotation vector and translation vector
    from a 4x4 transformation matrix.
    
    Parameters:
        T (np.ndarray): 4x4 transformation matrix

    Returns:
        rot (np.ndarray): Rotation vector (3,)
        trans (np.ndarray): Translation vector (3,)
    """
    # Extract rotation matrix and translation vector
    R_mat = T[:3, :3]
    trans = T[:3, 3]

    # Convert rotation matrix to rotation vector
    rot = Rotation.from_matrix(R_mat).as_rotvec()

    return rot, trans


def create_o3d_from_numpy(np_points, np_colors):
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(np_points)

    if isinstance(np_colors, list):
        np_colors = np.array(np_colors)
    if len(np_colors.shape) == 1:
        np_colors = np_colors[None]
    if len(np_colors) == 1:
        res.paint_uniform_color(np_colors[0])
    else:
        res.colors = o3d.utility.Vector3dVector(np_colors)
        
    return res

def remove_edges_from_depth(depth_np):
    """
    Removes edges from a depth numpy array using Canny edge detection and dilation.
    Returns a new depth array with edges set to zero.
    """
    depth_uint16 = depth_np.astype(np.uint16)
    depth_uint8 = np.empty_like(depth_uint16, dtype=np.uint8)
    cv2.normalize(depth_uint16, depth_uint8, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(depth_uint8, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    depth_processed = depth_np.copy()
    depth_processed[edges_dilated > 0] = 0

    # # --- Visualization of detected edges (for debugging) ---
    # # Uncomment the following lines to visualize edges side by side and scaled down
    # edges_combined = np.hstack((edges, edges_dilated))
    # scale_factor = 0.5  # Adjust as needed (e.g., 0.5 for half size)
    # edges_combined_small = cv2.resize(edges_combined, (0, 0), fx=scale_factor, fy=scale_factor)
    # cv2.imshow('Canny Edges (Left) | Canny Edges Dilated (Right)', edges_combined_small)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # ------------------------------------------------------

    return depth_processed

def pc_from_rgbd(path_rgb, path_depth, pose, intrinsics=None, crop=None):
    if intrinsics is None:
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # Load images
    color_bgr = cv2.imread(path_rgb)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Convert depth to numpy array for processing
    depth_raw = o3d.io.read_image(path_depth)
    depth_np = np.asarray(depth_raw)

    if crop is not None:
        depth_np = center_crop(depth_np, crop)
        color_rgb = center_crop(color_rgb, crop)

    # Remove edges from depth
    depth_np = remove_edges_from_depth(depth_np)

    color_raw = o3d.geometry.Image(np.ascontiguousarray(color_rgb))    

    # Convert cleaned depth back to Open3D image
    cleaned_depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, cleaned_depth, convert_rgb_to_intensity=False)

    # Create point cloud
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointcloud.transform(pose)

    return pointcloud

# --------------------------------------------
# Backproject only masked pixels, assign uniform color
# --------------------------------------------
def pc_from_rgbd_with_mask(color_path, depth_path, mask, pose,
                           intrinsic=o3d.camera.PinholeCameraIntrinsic(
                               o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                           depth_scale=1000.0,
                           depth_trunc=3.0):
    # Load images
    color = np.asarray(o3d.io.read_image(color_path))
    depth = np.asarray(o3d.io.read_image(depth_path)).astype(np.float64) 

    # Remove edges from depth
    depth = remove_edges_from_depth(depth)
    depth = depth / depth_scale

    # Mask & valid depth
    valid = mask & (depth > 0) & (depth < depth_trunc)
    coords = np.argwhere(valid)
    if coords.size == 0:
        return o3d.geometry.PointCloud()
    vs, us = coords[:, 0], coords[:, 1]
    zs = depth[vs, us]

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    pts = np.stack([xs, ys, zs], axis=1)

    # Transform to world
    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_world = (R @ pts.T).T + t

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    return pcd

def draw_pointclouds(pointclouds, trajs):
    if  isinstance(pointclouds, o3d.geometry.PointCloud):
        pointclouds = [pointclouds]

    elif not isinstance(pointclouds, list) or not isinstance(pointclouds[0], o3d.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud object.")
    
    o3d.visualization.draw_geometries(
        pointclouds,
        window_name="Open3D Point Cloud Viewer",
        width=800,
        height=600,
        point_show_normal=False
    )

def merge_pointclouds(pc_list, voxel_th=None):
    '''
    Merges a list of Open3D PointCloud objects into a single pointcloud.
    Optionally downsamples the resulting pointcloud using voxel_th (in meters).
    
    Args:
        pc_list (list): List of open3d.geometry.PointCloud instances.
        voxel_th (float, optional): Voxel size for downsampling.
    
    Returns:
        open3d.geometry.PointCloud: Merged (and optionally downsampled) point cloud.
    '''
    if not pc_list:
        raise ValueError("pc_list is empty.")

    merged_pc = o3d.geometry.PointCloud()
    
    for pc in pc_list:
        if not isinstance(pc, o3d.geometry.PointCloud):
            raise TypeError("All items in pc_list must be Open3D PointCloud objects.")
        merged_pc += pc

    if voxel_th is not None:
        merged_pc = merged_pc.voxel_down_sample(voxel_size=voxel_th)

    return merged_pc


def read_pointclouds(source_idx, target_idx, depth_base_path="lab1/data/livingroom1-depth-clean"):
    color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": source_idx})
    depth_raw = o3d.io.read_image(depth_base_path + '/%(number)05d.png'%{"number": source_idx})
    rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": target_idx})
    depth_raw = o3d.io.read_image(depth_base_path + '/%(number)05d.png'%{"number": target_idx})
    rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    source = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image0,
            o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image1,
            o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return source, target, rgbd_image0, rgbd_image1

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])


def visualize_point_cloud(pc):
  o3d.visualization.draw_geometries([pc],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
  
# Functions to read files containing the ground truth camera poses
class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)
    
    @property
    def rotation(self):
        return self.pose[:3, :3]
    
    @property
    def rotation_axis_angle(self):
        return Rotation.from_matrix(self.pose[:3, :3]).as_rotvec()
    
    @property
    def translation(self):
        return self.pose[:3, 3]

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
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

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, voxel_size, transform_ransac):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transform_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result
