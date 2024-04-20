import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.sceneobjects.targets.knifeedgetarget import KnifeEdgeTarget
from pycvsim.core.image_utils import overlay_points_on_image
from pycvsim.routines.knifeedge.edge import Edge
import panda3d
import panda3d.core
import cv2

class KnifeEdgeRoutine:
    camera: SceneCamera
    angle: float

    def __init__(self, camera: SceneCamera, angle: float = 8.4):
        self.camera = camera
        self.angle = angle
        self.target = KnifeEdgeTarget(0.8, angle=angle)
        self.renderer = Open3DRenderer(cameras=[self.camera], objects=[self.target])


    def run(self):
        # compute the edge points
        edge_object_points = self.target.get_edge_points()
        edge_image_points = self.camera.get_pixel_point_lies_in(edge_object_points)
        p0 = edge_image_points[0]
        p1 = edge_image_points[1]
        # compute a mask that correlates to the line
        mask = np.zeros((self.camera.yres, self.camera.xres), dtype=np.uint8)
        cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, thickness=3)

        image = self.renderer.render_image(camera_index=0, n_samples=1, return_as_8_bit=False)
        image[mask > 0] = self.renderer.render_image(camera_index=0, n_samples=500**2, mask=mask, return_as_8_bit=False)[mask > 0]

        edge = Edge(image, p0, p1)
        esf_x, esf_f = edge.get_edge_profile(normalise=False, search_region=4)
        plt.scatter(esf_x, esf_f)
        plt.show()