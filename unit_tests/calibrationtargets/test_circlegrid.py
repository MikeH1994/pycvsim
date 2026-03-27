import numpy as np
import open3d as o3d
from pycvsim.targets.utils import combine_submeshes
from pycvsim.targets.circlegrid.utils import create_square_with_circle_cutout, create_circle, create_circle_in_square


def run():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    run()
