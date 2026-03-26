from pycvsim.targets.calibrationtarget import CalibrationTarget
import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .utils import  create_circle_grid
import pycv

class CircleGridTarget(CalibrationTarget):
    def __init__(self, board_size, grid_width, radius, colour_circle=(0,0,0), colour_bkg=(255,255,255),
                 grid_height=None):
        grid_height = grid_height if grid_height is not None else grid_width
        color_circle = np.array(colour_circle)/255.0
        color_bkg = np.array(colour_bkg)/255.0
        vertices, triangles, colours = create_circle_grid(board_size, grid_width, grid_height, radius, color_circle, color_bkg)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
        object_points = pycv.calibration.CalibrationTarget(board_size, grid_width, grid_height).object_points
        super().__init__(mesh, object_points, None)
