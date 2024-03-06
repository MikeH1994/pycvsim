import numpy as np
from pycvsim.scene_objects.checkerboard_target import CheckerbordTarget
from pycvsim.scene_objects.scene_object import SceneObject
from pycvsim.rendering.scene_offscreen_renderer import SceneOffscreenRenderer
from pycvsim.rendering.scene_camera import SceneCamera
import matplotlib.pyplot as plt

mesh, object_points = CheckerbordTarget.create_target((7, 6), (0.05, 0.05), board_thickness=0.02,
                                                      color_bkg=(128, 0, 0), board_boundary=0.05)
#tensor_mesh, object_points = CircleGridTarget.create_target((7, 6), (0.05, 0.05), radius=0.015, color_bkg=(128, 0, 0),
#                                                     board_boundary=0.05)

# o3d.visualization.draw_geometries(objects_to_draw)

cameras = [
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, -1.5]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(640, 512), hfov=30.0),
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, -1.5]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(320, 256), hfov=120.0)
]

obj_mesh = SceneObject(mesh)
renderer = SceneOffscreenRenderer(cameras=cameras)
renderer.add_object(obj_mesh)

while True:
    for i in range(len(cameras)):
        img_1 = renderer.render_image(i)
        print("foo")
        plt.figure()
        plt.imshow(img_1)
    plt.show()
