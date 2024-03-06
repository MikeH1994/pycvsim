from .basecalibrationtarget import BaseCalibrationTarget
import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .utils import create_box, create_cylinder
import math
import matplotlib.pyplot as plt


class CircleGridTarget(BaseCalibrationTarget):
    def __init__(self, board_size, board_width):
        mesh, calibration_object_points = CircleGridTarget.create_target(board_size, width)
        super().__init__(mesh, calibration_object_points)

    @staticmethod
    def create_target(board_size, distance_between_points, radius,
                      color_circle=(255, 255, 255),
                      color_bkg=(255, 255, 255),
                      board_boundary=0.0) -> Tuple[o3d.t.geometry.TriangleMesh, NDArray]:
        color_circle = np.array(color_circle)/255.0
        color_bkg = np.array(color_bkg)/255.0
        board_width, board_height = board_size
        dx = distance_between_points
        object_points = np.zeros((board_height + 2, board_width + 2, 3))

        for i in range(board_width+2):
            for j in range(board_height+2):
                object_points[j, i] = [(i-1)*dx, (j-1)*dx, 0.0]

        mesh = o3d.geometry.TriangleMesh()

        for i in range(board_width+1):
            for j in range(board_height+1):
                mesh += create_cylinder(object_points[j, i], grid_width, grid_height, board_thickness, color)

        object_points = object_points[1:-1, 1:-1].reshape(-1, 3)

        centre = np.mean(object_points, axis=0)
        object_points -= centre

        mesh = mesh.translate(-centre)

        return mesh, object_points

