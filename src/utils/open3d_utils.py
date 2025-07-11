import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation

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

def pc_from_rgbd(path_rgb, path_depth, pose):
    color_raw = o3d.io.read_image(path_rgb)
    depth_raw = o3d.io.read_image(path_depth)
    rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image0,
            o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
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
    depth = np.asarray(o3d.io.read_image(depth_path)).astype(np.float64) / depth_scale

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

def draw_pointclouds(pointclouds):
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

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

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
