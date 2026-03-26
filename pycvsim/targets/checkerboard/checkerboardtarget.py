from pycvsim.targets.calibrationtarget import CalibrationTarget
import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from pycvsim.targets.utils import create_box
import pycv


class CheckerboardTarget(CalibrationTarget):
    def __init__(self, board_size: Tuple[int, int], grid_size: Tuple[float, float], board_thickness=0.05,
                 boundary_size: Tuple[float, float] = None):
        tensor_mesh = CheckerboardTarget.create_target(board_size, grid_size, board_thickness, boundary_size=boundary_size)
        obj_points = pycv.CalibrationTarget(board_size, *grid_size).get_object_points()
        super().__init__(tensor_mesh, obj_points)

    @staticmethod
    def create_target(board_size: Tuple[int, int], grid_size: Tuple[float, float], board_thickness: float,
                      boundary_size: Tuple[float, float] = None) -> o3d.t.geometry.TriangleMesh:
        color_1 = (255, 255, 255)
        color_2 = (0, 0, 0)
        color_bkg = (128, 128, 128)
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
        if boundary_size is not None:
            w, h = boundary_size
            kx, ky = (1+board_width) * grid_width, (1+board_height)*grid_height
            # top
            mesh += create_box(object_points[0, 0] - np.array([w, h, 0]), kx+2*w, h, board_thickness, color_bkg)
            # bottom
            mesh += create_box(object_points[-1, 0] - np.array([w, 0, 0]), kx + 2*w, h, board_thickness, color_bkg)
            # left
            mesh += create_box(object_points[0, 0] - np.array([w, h, 0]), w, ky + 2*h, board_thickness, color_bkg)
            # right
            mesh += create_box(object_points[0, -1], w, ky + h, board_thickness, color_bkg)

        return mesh
