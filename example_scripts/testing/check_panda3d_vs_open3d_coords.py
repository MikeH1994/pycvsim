import numpy as np
from pycvsim.sceneobjects.calibrationtargets.checkerboardtarget import CheckerbordTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera
import matplotlib.pyplot as plt
import cv2

"""obj = CheckerbordTarget((7, 7), (0.05, 0.05), board_thickness=0.02,
                        color_1 = (255, 255, 255), color_2 = (0, 0, 0),
                        color_bkg=(128, 0, 0), board_boundary=0.05, name="checkerboard")
"""
obj = SceneObject.load_armadillo()

cameras = [
    #SceneCamera.create_camera_from_euler_angles(object_pos=np.array([0.0, 0.0, -1.5]),
    #                                            euler_angles=np.array([0, 0, 0]),
    #                                            res=(720, 720), hfov=60.0)
    SceneCamera.create_camera_from_lookpos(pos=np.array([0.0, 0.0, -2.0]),
                                           lookpos=np.array([0.0, 0.0, 0.0]),
                                           up=np.array([0.0, 1.0, 0.0]),
                                           res=(720, 720), hfov=60.0, safe_zone=0)
]

renderer = SceneRenderer(cameras=cameras, objects=[obj])

#fig = plt.figure()
#ax_1 = fig.add_subplot(111)
#im_1 = ax_1.imshow(np.zeros((640, 512), dtype=np.uint8))
#plt.ion()

while True:
    for i in range(3):
        angles = np.random.uniform(low=-50, high=50, size=3)
        object_pos = np.random.uniform(low=-0.2, high=0.2, size=3)
        obj.set_pos(object_pos)
        obj.set_euler_angles(angles)

        lookpos = object_pos + np.random.uniform(low=-0.5, high=0.5, size=3)
        camera_pos = np.array([0.0, 0.0, -2.0]) + np.random.uniform(low=-0.5, high=0.5, size=3)
        renderer.set_camera_position(0, camera_pos)
        renderer.set_camera_lookpos(0, object_pos, np.array([0.0, 1.0, 0.0]))

        img_render = renderer.render_image(0, apply_distortion=True)
        img_1 = cv2.cvtColor(img_render, cv2.COLOR_RGB2GRAY)
        img_1 = (img_1 != 51).astype(np.uint8)
        img_2 = renderer.raycast_scene(0)["object_ids"] + 1
        img_3 = np.zeros((*img_1.shape, 3), dtype=np.uint8)
        img_3[img_1 == 1] = [255.0, 0.0, 0.0]
        img_3[img_2 == 1] = [0.0, 255.0, 0.0]
        img_3[(img_1 == 1) & (img_2 == 1)] = [255.0, 255.0, 0.0]

        plt.imshow(img_render)
        plt.figure()
        plt.imshow(img_2)
        plt.figure()
        plt.imshow(img_3)
        plt.show()


        """im_1.set_data(img_1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)"""
