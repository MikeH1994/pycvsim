from pycvsim.sceneobjects.sceneobject import SceneObject
import open3d as o3d
from numpy.typing import NDArray
import numpy as np
from typing import Tuple, Union
from pycvsim.targets.utils import transform_object_points
from pycvsim.rendering.open3drenderer import Open3DRenderer
import math


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

    @staticmethod
    def create_image(res: Tuple[int, int], angle: float, center: Tuple[float, float] = None,
                     color_1=(255, 255, 255), color_2=(0, 0, 0), n_samples=100, mode="fast"):
        color_1 = np.array(color_1)
        color_2 = np.array(color_2)
        dst_image = np.zeros((res[1], res[0], 3), dtype=np.uint8)

        cx, cy = center
        line = Line(center, angle)

        if mode == "default":
            for x in range(res[0]):
                for y in range(res[1]):
                    frac = line.sample_points(x, y, n_samples)
                    dst_image[y][x] = frac*color_1 + (1.0-frac)*color_2
        else:
            xx, yy = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
            frac = line.sample_points(xx, yy).reshape(*xx.shape, 1)
            dst_image = frac*color_1 + (1.0-frac)*color_2
        return dst_image


class Line:
    def __init__(self, centre, angle):
        self.centre = np.array(centre)
        self.angle = angle
        self.is_vertical = np.abs(angle%180.0) < 1e-6
        self.m = np.tan(np.radians(angle)) if not self.is_vertical else None
        self.c = self.centre[1] - self.centre[0] * self.m if not self.is_vertical else None

    def distance_to_line(self, x, y):
        # if line is vertical
        if self.is_vertical:
            return x - self.centre[0]
        else:
            return (y - (self.m * x + self.c)) / np.sqrt(self.m ** 2 + 1)

    def above_line(self, x, y):
        # if line is vertical
        if self.is_vertical:
            return x > self.centre[0]
        else:
            y_line = self.m*x + self.c
            return y < y_line

    def sample_points(self, x: Union[NDArray, float], y: Union[NDArray, float], n_samples=100):
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        if isinstance(y, float) or isinstance(y, int):
            y = np.array([y])
        assert(x.shape == y.shape)
        init_shape = x.shape
        x = x.reshape(-1)
        y = y.reshape(-1)

        n_samples = int(round(math.sqrt(n_samples))**2)

        multisamples = Open3DRenderer.get_multisample_pattern(n_samples)
        x_samples = np.zeros((x.shape[0], n_samples), dtype=np.float32)
        x_samples[:] = x.reshape(-1, 1) + multisamples[:, 0]
        y_samples = np.zeros((x.shape[0], n_samples), dtype=np.float32)
        y_samples[:] = y.reshape(-1, 1) + multisamples[:, 1]

        n = self.above_line(x_samples, y_samples).astype(np.int32)
        return np.mean(n, axis=-1).reshape(init_shape)