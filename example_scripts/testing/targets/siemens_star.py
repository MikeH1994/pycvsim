import numpy as np
from pycvsim.targets.siemensstar import SiemensStar
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
import matplotlib.pyplot as plt

res = (720, 640)
mesh = SiemensStar(radius=0.5, n_spokes=36)
cameras = [SceneCamera(pos=np.array([0.0, 0.0, -3]), res=res, hfov=30.0)]
renderer = Open3DRenderer(cameras=cameras, objects=[mesh])

while True:
    for theta in np.linspace(-180, 180, 100):
        mesh.set_euler_angles(np.array([0, 0, theta]))
        img_1 = renderer.render(0, n_samples=1**2)
        points = renderer.cameras[0].get_pixel_point_lies_in(mesh.get_object_points())
        img_1 = overlay_points_on_image(img_1, points, radius=6)
        points = renderer.cameras[0].get_pixel_point_lies_in(np.array([mesh.get_pos()]))
        img_1 = overlay_points_on_image(img_1, points,
                                        color=[0, 255, 0], radius=6)
        plt.imshow(img_1)
        plt.show()