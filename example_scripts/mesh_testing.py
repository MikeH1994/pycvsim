import numpy as np
import matplotlib.pyplot as plt
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.scenerenderer import SceneRenderer
from pycvsim.rendering.scenecamera import SceneCamera


cameras = [
    SceneCamera.create_camera_from_euler_angles(pos=np.array([0.0, 0.0, 0.0]),
                                                euler_angles=np.array([0, 0, 0]),
                                                res=(640, 512), hfov=50.0),
]

renderer = SceneRenderer(cameras=cameras)
obj_mesh = SceneObject.load_armadillo()
obj_mesh.set_pos(np.array([0.4, 0.0, 3]))
renderer.add_object(obj_mesh)
obj_mesh = SceneObject.load_armadillo()
obj_mesh.set_pos(np.array([-0.4, 0.0, 3]))
renderer.add_object(obj_mesh)


while True:
    img = renderer.render_image(0)
    img_2 = renderer.raycast_scene(0)["object_ids"]
    plt.imshow(img)
    plt.figure()
    plt.imshow(img_2)
    plt.show()
