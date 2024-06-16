import open3d as o3d
import numpy as np
from typing import List
from numpy.typing import NDArray
from pycvsim.sceneobjects.utils import tensor_mesh_to_legacy_mesh


def create_box(bottom_left: NDArray, width: float, height: float,
               depth: float, color: NDArray):
    box = o3d.geometry.TriangleMesh.create_box(width, height, depth)

    vertex_colors = np.array([color for i in range(len(box.vertices))])
    box.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    box = box.translate(bottom_left)
    return box


def create_cylinder(position, radius, depth, color):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=depth)
    r = cylinder.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    vertex_colors = np.array([color for i in range(len(cylinder.vertices))])
    cylinder.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    cylinder = cylinder.rotate(r, center=(0, 0, 0))
    cylinder = cylinder.translate(position)
    cylinder = cylinder.translate((0, 0, depth/2))

    return cylinder

def create_sector_of_circle(theta_0: float, theta_1: float, radius: float, n_segments: int = 5, centre=np.zeros(3),
                            colour: NDArray = np.array([255.0, 255.0, 255.0])):
    def create_segment(index_0_, index_1_, triangle_indices_: List = None):
        #if (index_1_ - index_0_) % 2 == 1:
        #    raise Exception("foo")
        if triangle_indices_ is None:
            triangle_indices_ = []
        diff = (index_1_ - index_0_) // 2
        index_mid = index_0_ + diff
        triangle_indices_ += [[index_0_, index_1_, index_mid]]
        if diff <=2:
            return triangle_indices_
        triangle_indices_ = create_segment(index_0_, index_mid, triangle_indices_)
        triangle_indices_ = create_segment(index_mid, index_1_, triangle_indices_)
        return triangle_indices_

    n_points = (2**n_segments + 1)
    vertices = np.full((n_points, 3), fill_value=centre)
    theta = np.linspace(theta_0, theta_1, vertices.shape[0]-1)
    vertices[1:, 0] += radius*np.cos(theta)
    vertices[1:, 1] += radius*np.sin(theta)
    vertex_colours = np.full(vertices.shape, fill_value=colour, dtype=np.float32) / 255.0
    triangle_indices = np.array(create_segment(1, n_points-1, [[0, 1, n_points-1]]), dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colours)
    mesh.triangles = o3d.utility.Vector3iVector(triangle_indices)
    return mesh


def transform_object_points(object_points: NDArray, translation: NDArray, euler_angles: NDArray, degrees=True):
    if degrees:
        euler_angles = np.radians(euler_angles)
    init_shape = object_points.shape
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(object_points.reshape(-1, 3))
    # rotate
    r = pcl.get_rotation_matrix_from_xyz(euler_angles)
    pcl = pcl.rotate(r, center=(0, 0, 0))
    # translate
    pcl = pcl.translate(translation)
    object_points_transformed = np.array(pcl.points).reshape(init_shape)
    return object_points_transformed
