from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.routines.stereophotogrammetry.utils import depth_to_disparity, disparity_to_depth
import cv2
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt


class StereoRoutine:
    def __init__(self, camera_1: SceneCamera, camera_2: SceneCamera):
        self.camera_1 = camera_1
        self.camera_2 = camera_2
        self.image_size = camera_1.image_size
        self.renderer = Open3DRenderer(cameras=[self.camera_1, self.camera_2])
        self.map_x1, self.map_y1 = None, None
        self.map_x2, self.map_y2 = None, None
        self.unrectified_to_rectified_transform = None
        self.rectified_to_unrectified_transform = None
        self.initialise()

    def initialise(self):
        """
        in opencv, T is given as a (3,1) matrixthat describes the translation from
        the device 2 to the origin (device 1). Similarly, R defines the rotation matrix to get from the direction of
        camera 2 to the origin direction (camera 1 direction). Invert to get the opposite
        """

        T = self.camera_1.pos - self.camera_2.pos
        R = np.matmul(np.linalg.inv(self.camera_2.r), self.camera_1.r)
        R1 ,R2 ,P1 ,P2 ,self.q ,_ ,_ = cv2.stereoRectify(self.camera_1.camera_matrix, self.camera_1.distortion_coeffs,
                                                         self.camera_2.camera_matrix, self.camera_2.distortion_coeffs,
                                                         self.image_size, R, T, None, None, flags=0, alpha=-1)

        # manually set c_x and c_y to be equal to the principal point of left camera in reprojection matrix (so that the 3D coordinate of the centre of the depth image corresponds to x=0,y=0)
        self.q[0][3] = -self.camera_1.camera_matrix[0][2]  # c_x
        self.q[1][3] = -self.camera_1.camera_matrix[1][2]  # c_y
        self.map_x1, self.map_y1 = cv2.initUndistortRectifyMap(self.camera_1.camera_matrix,
                                                               self.camera_1.distortion_coeffs,
                                                               R1, P1, self.image_size, cv2.CV_32F)
        self.map_x2, self.map_y2 = cv2.initUndistortRectifyMap(self.camera_2.camera_matrix,
                                                               self.camera_2.distortion_coeffs,
                                                               R2, P2, self.image_size, cv2.CV_32F)
        self.unrectified_to_rectified_transform = np.copy(R1)
        self.rectified_to_unrectified_transform = np.linalg.inv(R1)

    def run(self, mesh: SceneObject, block_size=15, speckle_window_size=31, apply_filter=False):
        self.renderer.remove_all_objects()
        self.renderer.add_object(mesh)

        mesh_points = mesh.mesh()
        distance_to_vertices = np.linalg.norm(mesh_points.vertices - self.camera_1.pos, axis=1)
        min_distance = np.min(distance_to_vertices)
        max_distance = np.max(distance_to_vertices)

        mask = self.renderer.raycast_scene(0)["mask"]

        plt.imshow(mask)
        plt.show()

        image_1 = self.renderer.render_image(0)
        image_2 = self.renderer.render_image(1)
        image_1 = cv2.remap(image_1, self.map_x1, self.map_y1, cv2.INTER_LINEAR)
        image_2 = cv2.remap(image_2, self.map_x2, self.map_y2, cv2.INTER_LINEAR)

        plt.imshow(image_1)
        plt.figure()
        plt.imshow(image_2)
        plt.show()

        n_channels = 3

        disp1 = int(depth_to_disparity(self.q, min_distance))
        disp2 = int(depth_to_disparity(self.q, max_distance))
        min_disp = min(disp1, disp2) - min(disp1, disp2) % 16
        max_disp = max(disp1, disp2) + (16 - max(disp1, disp2) % 16)

        num_disp = max_disp - min_disp
        sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp,
                                     blockSize=block_size)
        sgbm.setP1(2 * n_channels * block_size ** 2)
        sgbm.setP2(4 * n_channels * block_size ** 2)
        sgbm.setSpeckleWindowSize(speckle_window_size)
        sgbm.setSpeckleRange(3)

        if apply_filter:
            sigma = 0.6
            lmbda = 16000.0
            left_matcher = sgbm
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher);
            left_disp = left_matcher.compute(image_1, image_2)
            right_disp = right_matcher.compute(image_2, image_1)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            left_disp_filtered = wls_filter.filter(left_disp, image_1, disparity_map_right=right_disp);
            img_disp = left_disp_filtered.astype(np.float32) / 16.0
        else:
            img_disp = sgbm.compute(image_1, image_2).astype(np.float32) / 16.0

        img_disp[mask == 0] = min_disp
        img_depth = cv2.reprojectImageTo3D(img_disp, self.q)

        plt.imshow(img_disp)
        plt.figure()
        plt.imshow(img_depth[:, :, 2])
        plt.show()

        return img_depth
