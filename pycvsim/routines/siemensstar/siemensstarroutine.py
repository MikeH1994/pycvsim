from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.sceneobjects.targets.siemensstar import SiemensStar
import pycvsim.routines.stereophotogrammetry.utils as utils
import cv2
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import scipy.spatial.transform
from numpy.typing import NDArray
import scipy.interpolate


class SiemensStarRoutine:
    def __init__(self, camera: SceneCamera, target: SiemensStar):
        self.camera = camera
        self.target = target
        self.renderer = Panda3DRenderer(cameras=[self.camera], objects=[self.target]) # Open3DRenderer

    def run(self):
        image = self.renderer.render(0, n_samples=30**2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        plt.imshow(image)
        plt.show()

        height, width = image.shape[:2]
        x = np.arange(width)
        y = np.arange(height)
        interp_fn = scipy.interpolate.RegularGridInterpolator((y, x), image)
        p = self.camera.get_pixel_point_lies_in(self.target.get_object_points())
        center = self.camera.get_pixel_point_lies_in(self.target.get_pos())
        max_radius = np.min(np.sqrt((p[:, 0] - center[0])**2 + (p[:, 1] - center[1])**2))

        for l in [center[0], center[1], width - center[0], height - center[1]]:
            if max_radius > l:
                max_radius = l

        radii = list(np.linspace(1.0, max_radius*0.8))
        data = []
        for radius in radii:
            data.append(self.generate_line(interp_fn, radius))

        plt.plot(data[len(radii)//2])
        plt.show()

    def generate_line(self, interp_fn, radius: float, n_elems: int = 3000):
        theta = np.linspace(0, 2*np.pi, n_elems)
        cx, cy = self.camera.get_pixel_point_lies_in(self.target.get_pos())
        x_points = cx + radius * np.cos(theta)[:-1]
        y_points = cy + radius * np.sin(theta)[:-1]
        line = interp_fn((y_points, x_points))
        return line

