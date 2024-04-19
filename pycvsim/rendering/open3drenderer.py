from typing import List
import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pycvsim.rendering.baserenderer import BaseRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject
from .scenecamera import SceneCamera


class Open3DRenderer(BaseRenderer):
    def __init__(self, cameras: List[SceneCamera] = None, objects: List[SceneObject] = None, n_samples=1, background_colour=(0,0,0)):
        super().__init__(cameras=cameras, objects=objects)

    def render_image(self, camera_index, apply_distortion=True, n_samples=32, mask=None, return_as_8_bit=True,
                     background_colour=(51, 51, 51)):
        background_colour = np.array(background_colour)
        if camera_index >= len(self.cameras):
            raise Exception("Camera index {} is out of bounds".format(camera_index))

        camera = self.cameras[camera_index]
        rays = camera.generate_rays(apply_distortion=apply_distortion, n_samples=n_samples, mask=mask)
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        for obj in self.objects:
            raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh()))
        ans = raycasting_scene.cast_rays(o3d.core.Tensor(rays))

        object_ids = ans["geometry_ids"].numpy()
        primitive_ids = ans["primitive_ids"].numpy()
        offset = 0
        for i in range(len(self.objects)):
            primitive_ids[object_ids == i] -= offset
            offset += np.asarray(self.objects[i].original_mesh.triangles).shape[0]

        samples = np.zeros((*rays.shape[:-1], 3), dtype=np.float32)
        for i in range(len(self.objects)):
            # calculate the vertex colours for each triangle
            vertex_colours = np.array(self.objects[i].original_mesh.vertex_colors, dtype=np.float32)*255.0
            triangle_indices = np.asarray(self.objects[i].original_mesh.triangles)
            triangle_colours = vertex_colours[triangle_indices[:, 0]]
            triangle_colours += vertex_colours[triangle_indices[:, 1]]
            triangle_colours += vertex_colours[triangle_indices[:, 2]]
            triangle_colours /= 3
            samples[object_ids == i] = triangle_colours[primitive_ids[object_ids == i]]
        samples[object_ids == raycasting_scene.INVALID_ID] = background_colour
        samples = np.mean(samples, axis=-2)

        if mask is not None:
            dst_image = np.zeros((camera.yres, camera.xres, 3)) + background_colour
            dst_image[mask > 0] = samples
        else:
            dst_image = samples
        if return_as_8_bit:
            return dst_image.astype(np.uint8)
        return dst_image
