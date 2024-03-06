import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import List
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter, GeomNode, NodePath


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


"""def tensor_mesh_to_legacy_mesh(tensor_mesh: o3d.t.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    legacy_mesh = o3d.geometry.TriangleMesh()
    vertices = np.copy(np.array(tensor_mesh.vertex.positions, dtype=np.float32))
    triangles = np.copy(np.array(tensor_mesh.triangle.indices, dtype=np.int32))
    vertex_colors = np.copy(np.array(tensor_mesh.vertex.colors, dtype=np.float32))
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    legacy_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    legacy_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return tensor_mesh"""


def tensor_mesh_to_legacy_mesh(tensor_mesh: o3d.t.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    vertices = []
    vertex_colors = None
    triangle_indices = []

    for i in range(len(tensor_mesh.vertex.positions)):
        v = tensor_mesh.vertex.positions[i]
        vertices.append([v[0].item(), v[1].item(), v[2].item()])

    if 'colors' in tensor_mesh.vertex:
        vertex_colors = []
        for i in range(len(tensor_mesh.vertex.colors)):
            v = tensor_mesh.vertex.colors[i]
            vertex_colors.append([v[0].item(), v[1].item(), v[2].item()])

    for i in range(len(tensor_mesh.triangle.indices)):
        v = tensor_mesh.triangle.indices[i]
        triangle_indices.append([v[0].item(), v[1].item(), v[2].item()])

    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if vertex_colors is not None:
        legacy_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    legacy_mesh.triangles = o3d.utility.Vector3iVector(triangle_indices)
    return legacy_mesh



def o3d_mesh_to_pandas3d(mesh: o3d.geometry.TriangleMesh) -> NodePath:
    fmt = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('', fmt, Geom.UHDynamic)
    vertices = GeomVertexWriter(vdata, 'vertex')
    vertex_colors = GeomVertexWriter(vdata, 'color')
    triangle_indices = GeomTriangles(Geom.UHDynamic)

    for i in range(len(mesh.vertices)):
        vertex = mesh.vertices[i]
        color = mesh.vertex_colors[i]
        vertex_colors.addData4f(color[2], color[1], color[0], 1.0)
        vertices.addData3(vertex[0], vertex[1], vertex[2])

    for i in range(len(mesh.triangles)):
        triangle = mesh.triangles[i]
        triangle_indices.addVertices(triangle[0], triangle[1], triangle[2])

    geom = Geom(vdata)
    geom.addPrimitive(triangle_indices)

    geom_node = GeomNode('')
    geom_node.addGeom(geom)
    return geom_node