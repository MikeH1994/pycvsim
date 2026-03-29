from typing import List, Union
import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycv import PinholeCamera
from scipy.stats import qmc
import scipy.ndimage
import scipy.signal


class Renderer:
    def __init__(self, cameras: List[PinholeCamera] = None, objects: List[SceneObject] = None):
        cameras = cameras if cameras is not None else []
        objects = objects if objects is not None else []
        self.objects: List[SceneObject] = []
        self.cameras: List[PinholeCamera] = []
        self._cached_patterns = {}
        for camera in cameras:
            self.add_camera(camera)
        for obj in objects:
            self.add_object(obj)

    def add_camera(self, camera: PinholeCamera):
        self.cameras.append(camera)

    def remove_object(self, object_index):
        self.objects.pop(object_index)

    def add_object(self, obj: SceneObject):
        self.objects.append(obj)

    def remove_all_objects(self):
        while len(self.objects) > 0:
            self.remove_object(0)

    def render(self, camera_index, return_as_8_bit=True,
               n_samples: Union[List, int]=1, n_bkg_samples: int = 1, max_batch_size_mb = 2048):
        """

        :param camera_index:
        :param kwargs:
        :return:
        """
        if camera_index >= len(self.cameras):
            raise Exception("Camera index {} is out of bounds".format(camera_index))
        camera = self.cameras[camera_index]

        w, h = camera.res()
        if isinstance(n_samples, int):
            n_samples = [(n_samples, np.ones((h, w), dtype=np.uint8))]
        else:
            mask = np.ones((h, w), dtype=np.uint8)
            n_samples.insert(0, (n_bkg_samples, mask))

        image = np.zeros((h, w, 3))
        for (n_samps, samples_mask) in n_samples:
            img_ = self._render_(camera, return_as_8_bit=False, n_subsamples=n_samps, mask=samples_mask, max_batch_size_mb=max_batch_size_mb)
            image[samples_mask > 0] = img_[samples_mask > 0]

        if return_as_8_bit:
            return image.astype(np.uint8)

        return image

    def _render_(self, camera: PinholeCamera, n_subsamples=32, mask=None, return_as_8_bit=True,
                 background_colour=np.array([51.0, 51.0, 51.0]), max_batch_size_mb=1024):
        """

        :param camera:
        :param n_subsamples:
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
        n_pixels = x_pixels.shape[0]
        batch_size = int(max_batch_size_mb * 1e6 / (8 * n_subsamples))

        for i in range(0, n_pixels, batch_size):
            y_pixels_i = y_pixels[i:i+batch_size+1]
            x_pixels_i = x_pixels[i:i+batch_size+1]
            if len(y_pixels_i) == 0 or len(x_pixels_i) == 0:
                continue
            samples = self._render_samples(raycasting_scene, camera, x_pixels_i, y_pixels_i,
                                           n_subsamples=n_subsamples, background_colour=background_colour)

            dst_image[y_pixels_i, x_pixels_i, :] = samples

        if return_as_8_bit:
            return dst_image.astype(np.uint8)
        return dst_image

    def _render_samples(self, raycasting_scene: o3d.t.geometry.RaycastingScene, camera: PinholeCamera,
                        x: NDArray, y: NDArray, n_subsamples=1,
                        background_colour: NDArray = np.array([51.0, 51.0, 51.0])):
        """

        :param raycasting_scene:
        :param camera:
        :param x:
        :param y:
        :param n_subsamples:
        :param background_colour:
        :return:
        """

        # create an array containing the x, y coordinates of the pixels we are going to sample
        n_pixels = x.shape[0]
        pixel_samples = np.zeros((n_pixels, n_subsamples, 2), dtype=np.float32)
        pixel_samples[:, :, 0] = x.reshape(-1, 1)
        pixel_samples[:, :, 1] = y.reshape(-1, 1)

        # in this pixels, add the multisampling pattern
        pixel_samples += self.get_multisample_pattern(n_subsamples, x, y)

        # print(f"size of samples {pixel_samples.size * pixel_samples.itemsize / (1024 ** 2)} mb")

        # generate rays and the raycast
        rays = camera.generate_rays(apply_undistortion=True, pixel_coords=pixel_samples)
        ans = raycasting_scene.cast_rays(o3d.core.Tensor(rays))

        object_ids = ans["geometry_ids"].numpy()
        primitive_ids = ans["primitive_ids"].numpy()
        uv = ans["primitive_uvs"].numpy()

        # by default, open3d has each primitive assigned a unique id
        # for us it is more convenient if the ID represents the index of the corresponding vertices
        offset = 0
        for i in range(len(self.objects)):
            primitive_ids[object_ids == i] -= offset
            offset += np.asarray(self.objects[i].get_mesh().triangles).shape[0]

        # create an array of shape (a, b, c, ... , 3) to store sample results
        samples = np.zeros((*rays.shape[:-1], 3), dtype=np.float32)
        # loop through each object in the scene, and find the samples that hit this object
        #  w = 1 − u − v
        # color = w*C0 + u*C1 + v*C2
        for i in range(len(self.objects)):
            # calculate the vertex colours for each triangle

            vertex_colours = np.array(self.objects[i].mesh.vertex_colors, dtype=np.float32) * 255.0
            triangle_indices = np.asarray(self.objects[i].mesh.triangles)

            hits = (object_ids == i)
            hit_tris = primitive_ids[hits]  # (N_hit,) triangle indices
            u = uv[hits, 0]  # (N_hit,)
            v = uv[hits, 1]
            w = 1.0 - u - v

            # Fetch triangle vertex indices (N_hit, 3)
            tris = triangle_indices[hit_tris]

            # Vertex colors (float32 RGB, multiply by 255 if needed)
            C0 = vertex_colours[tris[:, 0]]  # (N_hit, 3)
            C1 = vertex_colours[tris[:, 1]]
            C2 = vertex_colours[tris[:, 2]]

            # Compute interpolated color
            samples[hits] = (
                    C0 * w[:, None] +
                    C1 * u[:, None] +
                    C2 * v[:, None]
            )
        # these samples do not hit anything
        samples[object_ids == raycasting_scene.INVALID_ID] = background_colour
        # for each pixel, take average over all samples
        samples = np.mean(samples, axis=1)
        return samples


    def get_multisample_pattern(self, n_samples, x, y):
        """
        Returns a per-pixel scrambled QMC sampling pattern in [-0.5, 0.5]^2.
        - A single global Halton pattern is generated per n_samples.
        - Each pixel applies a Cranley–Patterson rotation based on a hash of (x, y).
        """
        n_pixels = x.shape[0]

        # 1. Cache the base QMC pattern (one per n_samples)
        if n_samples not in self._cached_patterns:
            base = qmc.Halton(d=2, scramble=True, seed=12345).random(n_samples)
            self._cached_patterns[n_samples] = base  # in [0,1)

        base = self._cached_patterns[n_samples].reshape(1, -1, 2)

        # 3. Compute a deterministic scramble per pixel
        #    (fast 32‑bit integer hash → two floats in [0,1))
        h = self.pcg_hash(x, y)
        jx = ((h & 0xFFFF) / 65536.0)
        jy = ((h >> 16)  / 65536.0)
        jitter = np.stack((jx, jy), axis=1, dtype=np.float32).reshape(-1, 1, 2)

        # 4. Apply Cranley–Patterson rotation (mod 1) and shift to range[-0.5, 0.5]
        multisamples = np.zeros((n_pixels, n_samples, 2), dtype=np.float32)
        multisamples += (base + jitter) % 1.0 - 0.5
        return multisamples

    def pcg_hash(self, x, y):
        v = np.uint32(x) * 1664525 + np.uint32(y) * 1013904223
        v ^= v >> 16
        v *= 22695477
        v ^= v >> 16
        return v
