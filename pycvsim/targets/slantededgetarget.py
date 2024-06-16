from pycvsim.sceneobjects.sceneobject import SceneObject
import open3d as o3d
from numpy.typing import NDArray
import numpy as np
from typing import Tuple
from pycvsim.targets.utils import transform_object_points


class SlantedEdgeTarget(SceneObject):
    def __init__(self, width, angle=5.0, name=""):
        mesh, points = SlantedEdgeTarget.create(width, angle)
        self.edge_points = points
        super().__init__(mesh, name)

    def get_edge_points(self, transformed=True):
        if transformed:
            pos = self.get_pos()
            euler_angles = self.get_euler_angles()
            return transform_object_points(self.edge_points, pos, euler_angles)
        else:
            return np.copy(self.edge_points)

    @staticmethod
    def create(width, angle, color_1=(255, 255, 255), color_2=(0, 0, 0)) -> Tuple[o3d.t.geometry.TriangleMesh, NDArray]:
        color_1 = [i / 255.0 for i in color_1]
        color_2 = [i / 255.0 for i in color_2]
        tl = np.array([-width/2, width/2, 0.0])
        tr = np.array([width/2, width/2, 0.0])
        bl = np.array([-width/2, -width/2, 0.0])
        br = np.array([width/2, -width/2, 0.0])
        angle %= 360
        if angle < 45.0:
            p1 = np.array([-width/2.0*np.tan(np.radians(angle)), width/2.0, 0.0])
            p2 = -p1
            vertices = [tl, p1, bl, p1, bl, p2,
                        p1, tr, p2, tr, br, p2]
            vertex_colors = [color_2, color_2, color_2, color_2, color_2, color_2,
                             color_1, color_1, color_1, color_1, color_1, color_1]
        elif angle < 135.0:
            a = angle-90
            p1 = np.array([width/2.0, np.sign(a)*width/2.0*np.tan(np.radians(np.abs(a))), 0.0])
            p2 = -p1
            vertices = [tl, tr, p2, tr, p2, p1,
                        p1, p2, bl, p1, bl, br]
            vertex_colors = [color_1, color_1, color_1, color_1, color_1, color_1,
                             color_2, color_2, color_2, color_2, color_2, color_2]
        elif angle < 225.0:
            a = angle - 180
            p1 = np.array([np.sign(a)*width/2.0*np.tan(np.radians(np.abs(a))), -width/2.0, 0.0])
            p2 = -p1
            vertices = [tl, bl, p2, bl, p2, p1,
                        p1, p2, tr, p1, br, tr]
            vertex_colors = [color_1, color_1, color_1, color_1, color_1, color_1,
                             color_2, color_2, color_2, color_2, color_2, color_2]
        elif angle < 315.0:
            a = angle - 270
            p1 = np.array([-width/2.0, -np.sign(a)*width/2.0*np.tan(np.radians(np.abs(a))), 0.0])
            p2 = -p1
            vertices = [tl, p2, tr, tl, p2, p1,
                        p1, p2, br, br, p1, bl]
            vertex_colors = [color_2, color_2, color_2, color_2, color_2, color_2,
                             color_1, color_1, color_1, color_1, color_1, color_1]
        elif angle < 360.0:
            a = 360-angle
            p1 = np.array([width/2.0*np.tan(np.radians(a)), width/2.0, 0.0])
            p2 = -p1
            vertices = [tl, p1, bl, p1, bl, p2,
                        p1, tr, p2, tr, p2, br]
            vertex_colors = [color_2, color_2, color_2, color_2, color_2, color_2,
                             color_1, color_1, color_1, color_1, color_1, color_1]
        else:
            raise Exception()
        triangle_indices = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        mesh.triangles = o3d.utility.Vector3iVector(triangle_indices)

        return mesh, np.array([p1, p2])
