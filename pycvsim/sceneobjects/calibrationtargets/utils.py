import open3d as o3d
import numpy as np
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


def create_box_with_hole(bottom_left, hole_pos, square_width, square_height,
                         depth, circle_radius, square_color):
    square = create_box(bottom_left, square_width, square_height, depth, square_color)
    dx = 0.0 * depth
    hole_pos = [hole_pos[0], hole_pos[1], hole_pos[2] - dx/2]
    cutout_1 = create_cylinder(hole_pos, circle_radius, depth + dx, [0.0, 0.0, 0.0])
    cutout_2 = o3d.geometry.TriangleMesh.create_sphere(circle_radius).translate([hole_pos[0], hole_pos[1], hole_pos[2]])
    cutout_3 = o3d.geometry.TriangleMesh.create_sphere(circle_radius).translate([hole_pos[0], hole_pos[1],
                                                                                hole_pos[2] + depth])
    # o3d.visualization.draw_geometries([square, cutout_1])  # square

    # convert to tensor tensor_mesh so we can do boolean stuff
    # o3d.visualization.draw_geometries([square])  # square
    square = o3d.t.geometry.TriangleMesh.from_legacy(square)
    cutout_1 = o3d.t.geometry.TriangleMesh.from_legacy(cutout_1)
    cutout_2 = o3d.t.geometry.TriangleMesh.from_legacy(cutout_2)
    cutout_3 = o3d.t.geometry.TriangleMesh.from_legacy(cutout_3)

    #square = square.boolean_difference(cutout_1) # , tolerance=1.0
    #square = square.boolean_difference(cutout_2) # , tolerance=1.0
    square = square.boolean_difference(cutout_3) # , tolerance=1.0

    square = tensor_mesh_to_legacy_mesh(square)
    o3d.visualization.draw_geometries([square])  # square
    return square


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
