from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.open3drenderer import Open3DRenderer
from pycvsim.sceneobjects.sceneobject import SceneObject
import pycvsim.routines.stereophotogrammetry.utils as utils
import cv2
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import scipy.spatial.transform


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
        """
        https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectify.html
        
        Input parameters:
        cameraMatrix1 First camera matrix 3x3.
        distCoeffs1 First camera distortion parameters of 4, 5, 8, 12 or 14 elements.
        cameraMatrix2 Second camera matrix 3x3.
        distCoeffs2 Second camera distortion parameters of 4, 5, 8, 12 or 14 elements.
        imageSize Size of the image used for stereo calibration [w,h].
        R Rotation matrix between the coordinate systems of the first and the second cameras, 3x3/3x1 (see cv.Rodrigues)
        T Translation vector between coordinate systems of the cameras, 3x1.
        
        Output parameters:
        R1 3x3 rectification transform (rotation matrix) for the first camera.
        R2 3x3 rectification transform (rotation matrix) for the second camera.
        P1 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
        P2 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
        Q 4x4 disparity-to-depth mapping matrix (see cv.reprojectImageTo3D).
        roi1, roi2 rectangles inside the rectified unit_test_images where all the pixels are valid [x,y,w,h]. If Alpha=0, the ROIs cover the whole unit_test_images. Otherwise, they are likely to be smaller.
        """

        self.R1, self.R2, self.P1, self.P2, self.q, _, _ = cv2.stereoRectify(self.camera_1.camera_matrix,
                                                                             self.camera_1.distortion_coeffs,
                                                                             self.camera_2.camera_matrix,
                                                                             self.camera_2.distortion_coeffs,
                                                                             self.image_size, R, T, None, None,
                                                                             flags=0, alpha=0) # cv2.CALIB_ZERO_DISPARITY

        self.map_x1, self.map_y1 = cv2.initUndistortRectifyMap(self.camera_1.camera_matrix,
                                                               self.camera_1.distortion_coeffs,
                                                               self.R1, self.P1, self.image_size, cv2.CV_32F)
        self.map_x2, self.map_y2 = cv2.initUndistortRectifyMap(self.camera_2.camera_matrix,
                                                               self.camera_2.distortion_coeffs,
                                                               self.R2, self.P2, self.image_size, cv2.CV_32F)

    def get_images(self):
        image_1 = self.renderer.render(0, n_samples=1)
        image_2 = self.renderer.render(1, n_samples=1)
        image_1 = cv2.remap(image_1, self.map_x1, self.map_y1, cv2.INTER_LINEAR)
        image_2 = cv2.remap(image_2, self.map_x2, self.map_y2, cv2.INTER_LINEAR)
        return image_1, image_2

    def compute_disparity(self, image_1, image_2, min_disp, max_disp, block_size=3, speckle_window_size=31,
                          apply_filter=True):
        n_channels = 3
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
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            left_disp = left_matcher.compute(image_1, image_2)
            right_disp = right_matcher.compute(image_2, image_1)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            left_disp_filtered = wls_filter.filter(left_disp, image_1, disparity_map_right=right_disp);
            img_disp = left_disp_filtered.astype(np.float32) / 16.0
        else:
            img_disp = sgbm.compute(image_1, image_2).astype(np.float32) / 16.0
        return img_disp

    def generate_reconstruction(self, mesh: SceneObject, block_size=15, speckle_window_size=31, apply_filter=False):
        self.renderer.remove_all_objects()
        self.renderer.add_object(mesh)

        mesh_points = mesh.mesh()
        distance_to_vertices = np.linalg.norm(mesh_points.vertices - self.camera_1.pos, axis=1)
        min_distance = np.min(distance_to_vertices)
        max_distance = np.max(distance_to_vertices)

        mask = self.renderer.raycast_scene(0)["mask"]
        image_1, image_2 = self.get_images()
        min_disp, max_disp = utils.compute_min_and_max_disparity(self.q, min_distance, max_distance)
        img_disp = self.compute_disparity(image_1, image_2, min_disp=min_disp, max_disp=max_disp, block_size=block_size,
                                          speckle_window_size=speckle_window_size, apply_filter=apply_filter)
        img_disp[mask == 0] = min_disp
        mask[img_disp == min_disp] = 0
        mask[img_disp == max_disp] = 0
        img_reprojected = cv2.reprojectImageTo3D(img_disp, self.q)

        # go from rectified coordinate space in camera frame to non-rectified camera frame
        init_shape = img_reprojected.shape
        R = scipy.spatial.transform.Rotation.from_matrix(self.R1)
        img_reprojected = R.apply((img_reprojected).reshape(-1, 3), inverse=True).reshape(init_shape)

        # go from camera frame of reference to the world frame of reference
        # the camera matrix r defines the transformation from the
        R = scipy.spatial.transform.Rotation.from_matrix(self.camera_1.r)
        img_reprojected = R.apply((img_reprojected).reshape(-1, 3), inverse=False).reshape(init_shape)
        img_reprojected += self.camera_1.pos

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(img_reprojected[mask == 1].reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(image_1[mask == 1].reshape(-1, 3) / 255.0)

        return {
            "mask": mask,
            "img_1": image_1,
            "img_2": image_2,
            "disparity": img_disp,
            "depth": img_reprojected,
            "pointcloud": pcd
        }

    def run(self, mesh: SceneObject, block_size=15, speckle_window_size=31, apply_filter=False):
        res = self.generate_reconstruction(mesh, block_size, speckle_window_size, apply_filter)
        # R = scipy.spatial.transform.Rotation.from_matrix(self.camera_1.r)
        # mesh -= self.camera_1.pos
        plt.imshow(res["img_1"])
        plt.figure()
        plt.imshow(res["img_2"])
        plt.figure()
        plt.imshow(res["disparity"])
        plt.show()
        o3d.visualization.draw_geometries([res["pointcloud"], mesh.mesh()])
