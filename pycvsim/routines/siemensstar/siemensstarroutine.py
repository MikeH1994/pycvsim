from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.targets.siemensstar import SiemensStar
from scipy import signal
from scipy.signal import savgol_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform
import scipy.interpolate


class SiemensStarRoutine:
    def __init__(self, camera: SceneCamera, target: SiemensStar):
        self.camera = camera
        self.target = target
        self.renderer = Open3DRenderer(cameras=[self.camera], objects=[self.target]) # Open3DRenderer

    def run(self):
        image = self.renderer.render(0, n_samples=15**2)
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

        radii = list(np.linspace(5.0, max_radius*0.8))
        data = []
        for radius in radii:
            data.append(self.generate_line(interp_fn, radius))

        freq = [f["freq"] for f in data]
        m = [f["m"] for f in data]
        radius = [f["radius"] for f in data]
        plt.plot(radius, m)
        plt.show()
        #plt.plot(data[0]["theta"], data[0]["line"]) # len(radii)//2
        #plt.show()

    def generate_line(self, interp_fn, radius: float, n_elems: int = 3000):
        theta = np.linspace(0, 2*np.pi, n_elems)[:-1]
        cx, cy = self.camera.get_pixel_point_lies_in(self.target.get_pos())
        x_points = cx + radius * np.cos(theta)
        y_points = cy + radius * np.sin(theta)
        intensity = interp_fn((y_points, x_points))
        freq = self.target.n_spokes / (2 * np.pi * radius)

        yhat = savgol_filter(intensity, 11, 2) if freq < 0.45 else savgol_filter(intensity, 15, 2)

        maximums = scipy.signal.argrelextrema(yhat, np.greater, order=2)
        minimums = scipy.signal.argrelextrema(yhat, np.less, order=2)
        m = (intensity[maximums].mean() - intensity[minimums].mean()) / (intensity[maximums].mean() + intensity[minimums].mean())

        return {
            "freq": freq,
            "radius": radius,
            "x_points": x_points,
            "y_points": y_points,
            "theta": theta,
            "line": intensity,
            "m": m
        }
