from pycvsim.targets.calibrationtarget import CalibrationTarget
import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from pycvsim.targets.utils import create_box


class CheckerbordTarget(CalibrationTarget):
    def __init__(self, board_size: Tuple[int, int], grid_size: Tuple[float, float], board_thickness=0.05, name="", **kwargs):
        tensor_mesh, obj_points = CheckerbordTarget.create_target(board_size, grid_size, board_thickness, **kwargs)

        grid_width, grid_height = grid_size
        min_x, min_y, _ = np.min(obj_points, axis=0)
        max_x, max_y, _ = np.max(obj_points, axis=0)
        obj_boundary = np.array([
            np.array([min_x - grid_width, min_y - grid_height, 0.0]),
            np.array([max_x + grid_width, min_y - grid_height, 0.0]),
            np.array([min_x - grid_width, max_y + grid_height, 0.0]),
            np.array([max_x + grid_width, max_y + grid_height, 0.0])
        ])
        super().__init__(tensor_mesh, obj_points, obj_boundary, name)

    @staticmethod
    def create_target(board_size: Tuple[int, int], grid_size: Tuple[float, float], board_thickness: float,
                      **kwargs) -> Tuple[o3d.t.geometry.TriangleMesh, NDArray]:
        color_1 = (255, 255, 255)
        color_2 = (0, 0, 0)
        color_bkg = (255, 255, 255)
        for param in kwargs:
            if param == "color_2":
                color_1 = kwargs[param]
            elif param == "color_1":
                color_2 = kwargs[param]
            elif param == "color_bkg":
                color_bkg = kwargs[param]
            elif param == "board_boundary":
                board_boundary = kwargs[param]
            else:
                raise Exception("Unkown parameter {}".format(param))
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

        return mesh, object_points[::-1, :]
