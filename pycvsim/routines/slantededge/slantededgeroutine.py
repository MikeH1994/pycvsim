import numpy as np
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.targets.slantededgetarget import SlantedEdgeTarget
from pycvsim.core.utils import clip_line_to_image
import cv2
import scipy.ndimage


class SlantedEdgeRoutine:
    camera: BaseCamera
    angle: float

    def __init__(self, camera: BaseCamera, angle: float = 5.0):
        self.camera = camera
        self.angle = angle
        self.target = SlantedEdgeTarget(5.0, angle=angle)
        self.renderer = Open3DRenderer(cameras=[self.camera], objects=[self.target])

    def generate_image(self, n_samples=1024, fixed_multisample_pattern=False):
        # compute the edge points
        edge_object_points = self.target.get_edge_points()
        edge_image_points = self.camera.get_pixel_point_lies_in(edge_object_points)
        p0 = edge_image_points[0]
        p1 = edge_image_points[1]
        # compute a mask that correlates to the line
        mask = np.zeros((self.camera.yres, self.camera.xres), dtype=np.uint8)
        cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, thickness=3)
        n_samples = [(n_samples, mask)]

        image = self.renderer.render(camera_index=0, n_samples=n_samples, return_as_8_bit=False,
                                     fixed_multisample_pattern=fixed_multisample_pattern)
        p0, p1 = clip_line_to_image(p0, p1, (self.camera.xres, self.camera.yres))

        return image, p0, p1
