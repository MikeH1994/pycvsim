import open3d as o3d
import numpy as np
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter, GeomNode, NodePath


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
        if len(mesh.vertex_colors) == 0:
            color = [0.5, 0.5, 0.5]
        else:
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


def load_armadillo() -> o3d.geometry.TriangleMesh:
    # load the open3d armadillo mesh
    armadillo_mesh_data = o3d.data.ArmadilloMesh()
    armadillo_mesh = o3d.io.read_triangle_mesh(armadillo_mesh_data.path)
    # translate it then scale and rotate it so that it is centered on (0, 0, 0)
    armadillo_mesh = armadillo_mesh.translate(-armadillo_mesh.get_center())
    dx = 1.0 / (armadillo_mesh.get_max_bound() - armadillo_mesh.get_min_bound())[0]
    armadillo_mesh = armadillo_mesh.scale(dx, center=(0.0, 0.0, 0.0))
    r = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))
    armadillo_mesh = armadillo_mesh.rotate(r, armadillo_mesh.get_center())
    armadillo_mesh = apply_texturing_to_mesh(armadillo_mesh)
    return armadillo_mesh


def apply_texturing_to_mesh(mesh: o3d.geometry.TriangleMesh, mode="gradient"):
    shape = np.array(mesh.vertices).shape
    if mode == "random":
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.random.uniform(size=shape))
    elif mode == "gradient":
        points = np.asarray(mesh.vertices)
        vertex_colors = np.zeros(points.shape, dtype=np.float32)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
        vertex_colors[:, 0] = (points[:, 0]-min_x)/(max_x - min_x)
        vertex_colors[:, 1] = (points[:, 1]-min_y)/(max_y - min_y)
        vertex_colors[:, 2] = (points[:, 2]-min_z)/(max_z - min_z)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh
