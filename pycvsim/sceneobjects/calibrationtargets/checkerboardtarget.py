from pycvsim.sceneobjects.calibrationtargets.basecalibrationtarget import BaseCalibrationTarget
import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .utils import create_box


class CheckerbordTarget(BaseCalibrationTarget):
    def __init__(self, board_size, board_width, board_thickness, name=""):
        tensor_mesh, calibration_object_points = CheckerbordTarget.create_target(board_size, board_width, board_thickness)
        super().__init__(tensor_mesh, calibration_object_points, name)

    @staticmethod
    def create_target(board_size: Tuple[int, int], grid_size: Tuple[float, float], board_thickness: float,
                      color_1=(255, 255, 255), color_2=(0, 0, 0), color_bkg=(255, 255, 255),
                      board_boundary=0.0) -> Tuple[o3d.t.geometry.TriangleMesh, NDArray]:
        color_1 = np.array(color_1)/255.0
        color_2 = np.array(color_2)/255.0
        color_bkg = np.array(color_bkg)/255.0
        board_width, board_height = board_size
        grid_width, grid_height = grid_size
        object_points = np.zeros((board_height + 2, board_width + 2, 3))

        for i in range(board_width+2):
            for j in range(board_height+2):
                object_points[j, i] = [(i-1)*grid_width, (j-1)*grid_height, 0.0]

        mesh = o3d.geometry.TriangleMesh()

        for i in range(board_width+1):
            for j in range(board_height+1):
                color = color_1 if (i + j) % 2 == 0 else color_2
                mesh += create_box(object_points[j, i], grid_width, grid_height, board_thickness, color)
        object_points = object_points[1:-1, 1:-1].reshape(-1, 3)

        centre = np.mean(object_points, axis=0)
        object_points -= centre
        mesh = mesh.translate(-centre)

        return mesh, object_points


"""
        if board_boundary > 0.0:
            k = board_boundary
            bl = object_points[0, 0] + np.array([-dx, -dy, 0.0])
            br = object_points[0, board_width - 1] + np.array([dx, -dy, 0.0])
            tl = object_points[board_height - 1, 0] + np.array([-dx, dy, 0.0])
            tr = object_points[board_height - 1, board_width - 1] + np.array([dx, dy, 0.0])
            color = color_bkg

            # create boundary on top
            p0 = tl + np.array([-k, 0, 0.0])
            p1 = tr + np.array([k, 0, 0.0])
            p2 = tr + np.array([k, k, 0.0])
            p3 = tl + np.array([-k, k, 0.0])
            create_square([p0, p1, p2, p3], color, vertices=vertices, vertex_colors=vertex_colors,
                          triangle_indices=triangle_indices)

            # create boundary on bottom
            p0 = bl + np.array([-k, -k, 0.0])
            p1 = br + np.array([k, -k, 0.0])
            p2 = br + np.array([k, 0.0, 0.0])
            p3 = bl + np.array([-k, 0.0, 0.0])
            create_square([p0, p1, p2, p3], color, vertices=vertices, vertex_colors=vertex_colors,
                          triangle_indices=triangle_indices)

            # create boundary on left
            p0 = bl + np.array([-k, -k, 0.0])
            p1 = bl + np.array([0.0, -k, 0.0])
            p2 = tl + np.array([0.0, k, 0.0])
            p3 = tl + np.array([-k, k, 0.0])
            create_square([p0, p1, p2, p3], color, vertices=vertices, vertex_colors=vertex_colors,
                          triangle_indices=triangle_indices)

            # create boundary on right
            p0 = br + np.array([0.0, -k, 0.0])
            p1 = br + np.array([k, -k, 0.0])
            p2 = tr + np.array([k, k, 0.0])
            p3 = tr + np.array([0.0, k, 0.0])
            create_square([p0, p1, p2, p3], color, vertices=vertices, vertex_colors=vertex_colors,
                          triangle_indices=triangle_indices)
"""
