from pycvsim.targets.calibrationtarget import CalibrationTarget
import numpy as np
from numpy.typing import NDArray
from pycvsim.targets.utils import create_sector_of_circle


class SiemensStar(CalibrationTarget):
    def __init__(self, radius = 0.1, n_spokes = 30, colour_1: NDArray = np.array([255.0, 255.0, 255.0]),
                 colour_2: NDArray = np.array([0.0, 0.0, 0.0])):
        mesh, obj_points, obj_boundary = SiemensStar.create_target(radius, n_spokes, colour_1, colour_2)
        self.n_spokes = n_spokes
        super().__init__(mesh, obj_points, obj_boundary, "Siemens Star")

    @staticmethod
    def create_target(radius: float, n_spokes: int, colour_1: NDArray = np.array([255.0, 255.0, 255.0]),
                      colour_2: NDArray = np.array([0.0, 0.0, 0.0]), center: NDArray = np.zeros(3)):
        n_spokes *= 2
        theta = np.linspace(0, 2 * np.pi, n_spokes + 1)
        mesh = None
        boundary_points = np.array([[-radius, -radius, 0.0], [-radius, radius, 0.0],
                                    [radius, radius, 0.0], [radius, -radius, 0.0]], dtype=np.float32)
        for i in range(n_spokes):
            colour = colour_1 if i % 2 == 0 else colour_2
            mesh_i = create_sector_of_circle(theta[i], theta[i+1], radius, colour=colour, centre=center)
            if mesh is None:
                mesh = mesh_i
            else:
                mesh += mesh_i

        object_points = np.full((n_spokes, 3), fill_value=center)
        object_points[:, 0] += radius * np.cos(theta)[:-1]
        object_points[:, 1] += radius * np.sin(theta)[:-1]
        return mesh, object_points, boundary_points
