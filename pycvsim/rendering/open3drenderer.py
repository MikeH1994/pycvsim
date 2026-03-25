from typing import List
import math
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pycvsim.rendering.baserenderer import BaseRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycv import PinholeCamera
import psutil
from scipy.stats import qmc


class Open3DRenderer(BaseRenderer):
    def __init__(self, cameras: List[PinholeCamera] = None, objects: List[SceneObject] = None):
        super().__init__(cameras=cameras, objects=objects)

    def _render_(self, camera: PinholeCamera, n_samples=32, mask=None, return_as_8_bit=True,
                 background_colour=np.array([51.0, 51.0, 51.0]), fixed_multisample_pattern=True):
        """

        :param camera:
        :param n_samples:
        :param mask:
        :param return_as_8_bit:
        :param background_colour:
        :return:
        """
        background_colour = np.array(background_colour)

        xres, yres = camera.res()
        if mask is None:
            mask = np.ones((yres, xres), dtype=np.uint8)

        raycasting_scene = o3d.t.geometry.RaycastingScene()
        for obj in self.objects:
            raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj.get_mesh()))
        dst_image = np.full((yres, xres, 3), fill_value=background_colour, dtype=np.float32)
        y_pixels, x_pixels = np.where(mask == 1)
        i = 0
        k = 0.1
        while i <= y_pixels.shape[0]:
            available_memory = psutil.virtual_memory().available
            max_elems = int(k * available_memory / n_samples / 8)

            y_pixels_i = y_pixels[i:i+max_elems]
            x_pixels_i = x_pixels[i:i+max_elems]
            if len(y_pixels_i) == 0 or len(x_pixels_i) == 0:
                continue
            try:
                samples = self.render_samples(raycasting_scene, camera, x_pixels_i, y_pixels_i,
                                              n_samples=n_samples, background_colour=background_colour,
                                              fixed_multisample_pattern=fixed_multisample_pattern)
                dst_image[y_pixels_i, x_pixels_i, :] = samples
                i += max_elems
            except Exception:
                print("Open3D rendering buffer failed, reducing number of elements...")
                k /= 2

        if return_as_8_bit:
            return dst_image.astype(np.uint8)
        return dst_image

    def render_samples(self, raycasting_scene: o3d.t.geometry.RaycastingScene, camera: PinholeCamera,
                       x_indices: NDArray, y_indices: NDArray, n_samples=1,
                       background_colour: NDArray = np.array([51.0, 51.0, 51.0]), fixed_multisample_pattern=True):
        """

        :param raycasting_scene:
        :param camera:
        :param x_indices:
        :param y_indices:
        :param n_samples:
        :param background_colour:
        :return:
        """

        # create an array containing the x, y coordinates of the pixels we are going to sample
        pixels = np.zeros((x_indices.shape[0], n_samples, 2), dtype=np.float32)
        pixels[:, :, 0] = x_indices.reshape(-1, 1)
        pixels[:, :, 1] = y_indices.reshape(-1, 1)

        # in this pixels, add the multisampling pattern
        pixels += self.get_multisample_pattern(n_samples, fixed_multisample_pattern)

        # generate rays and the raycast
        rays = camera.generate_rays(apply_undistortion=True, pixel_coords=pixels)
        ans = raycasting_scene.cast_rays(o3d.core.Tensor(rays))

        object_ids = ans["geometry_ids"].numpy()
        primitive_ids = ans["primitive_ids"].numpy()

        # by default, open3d has each primitive assigned a unique id
        # for us it is more convenient if the ID represents the index of the corresponding vertices
        offset = 0
        for i in range(len(self.objects)):
            primitive_ids[object_ids == i] -= offset
            offset += np.asarray(self.objects[i].get_mesh().triangles).shape[0]


        # create an array of shape (a, b, c, ... , 3) to store sample results
        samples = np.zeros((*rays.shape[:-1], 3), dtype=np.float32)
        # loop through each object in the scene, and find the samples that hit this object
        for i in range(len(self.objects)):
            # calculate the vertex colours for each triangle
            vertex_colours = np.array(self.objects[i].mesh.vertex_colors, dtype=np.float32) * 255.0
            triangle_indices = np.asarray(self.objects[i].mesh.triangles)
            triangle_colours = vertex_colours[triangle_indices[:, 0]]
            triangle_colours += vertex_colours[triangle_indices[:, 1]]
            triangle_colours += vertex_colours[triangle_indices[:, 2]]
            triangle_colours /= 3
            samples[object_ids == i] = triangle_colours[primitive_ids[object_ids == i]]
        # these samples do not hit anything
        samples[object_ids == raycasting_scene.INVALID_ID] = background_colour
        # for each pixel, take average over all samples
        samples = np.mean(samples, axis=-2)
        return samples

    @staticmethod
    def get_multisample_pattern(n_samples: int = 1, fixed_pattern: bool = True):
        """

        :param n_samples:
        :return:
        """

        if fixed_pattern:
            pts = qmc.Halton(d=2, scramble=True).random(n_samples)
            return pts - 0.5
        else:
            multisamples = np.random.uniform(-0.5, 0.5, (n_samples, 2))
            return multisamples