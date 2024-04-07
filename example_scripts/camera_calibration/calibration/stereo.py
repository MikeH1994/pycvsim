import cv2
import numpy as np
import os
import copy
import pickle


class StereoPair:
    """
    R_2 - calibration 2 rotation from origin to this calibration- (openCV gives rotation matrix from this calibration to origin- invert before passing)
    T_2 - calibration 2 position relative to origin (openCV gives T translation calibration to origin- pass negative vector)

    """

    def __init__(self, device_1, device_2, object_points, dst_image_size=None,
                 R_1=np.eye(3), T_1=np.zeros(3)):
        self.device_1 = copy.deepcopy(device_1)
        self.device_2 = copy.deepcopy(device_2)

        self.image_size = None
        self.rms = 0
        self.R_1 = None
        self.R_2 = None
        self.T_1 = None
        self.T_2 = None
        self.Q = None
        self.map_x1 = None
        self.map_y1 = None
        self.map_x2 = None
        self.map_y2 = None
        self.calibration_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH

        self.initialise(dst_image_size)
        self.calibrate(object_points, R_1, T_1)
        print("Stereo pair {},{} calibrated - rms = {:.4f}".format(self.device_1.name,
                                                                   self.device_2.name,
                                                                   self.rms))

    def initialise(self, dst_image_size):
        if not dst_image_size is None:
            self.image_size = dst_image_size
        elif dst_image_size is None:
            n_pixels_1 = self.device_1.image_size[0] * self.device_1.image_size[1]
            n_pixels_2 = self.device_2.image_size[0] * self.device_2.image_size[1]
            if n_pixels_1 > n_pixels_2:
                self.image_size = self.device_2.image_size
            elif n_pixels_1 < n_pixels_2:
                self.image_size = self.device_1.image_size

        #self.device_1 = self.device_1.rescale_device(self.image_size)
        #self.device_2 = self.device_2.rescale_device(self.image_size)
        self.get_common_calibration_points()

    def open_img(self, folderpath, number, return_as_8_bit=True, invert_thermal=False):
        self.device_1.open_img(folderpath, number, return_as_8_bit=return_as_8_bit,
                               invert_thermal=invert_thermal)
        self.device_2.open_img(folderpath, number, return_as_8_bit=return_as_8_bit,
                               invert_thermal=invert_thermal)
        return self.device_1.img, self.device_2.img

    def get_common_calibration_points(self):
        self.stereo_image_points_1 = []
        self.stereo_image_points_2 = []
        for i1 in range(len(self.device_1.image_point_keys)):
            key = self.device_1.image_point_keys[i1]
            if key in self.device_2.image_point_keys:
                i2 = self.device_2.image_point_keys.index(key)
                self.stereo_image_points_1.append(self.device_1.image_points[i1])
                self.stereo_image_points_2.append(self.device_2.image_points[i2])

    def calibrate(self, object_points, R_1, T_1):
        assert (type(R_1) == np.ndarray and R_1.shape == (3, 3))
        assert (type(T_1) == np.ndarray and T_1.shape == (3,))
        n_calib_points = len(self.stereo_image_points_1)
        object_points = [object_points.astype(np.float32) for i in range(n_calib_points)]
        M1 = self.device_1.camera_matrix
        d1 = self.device_1.distortion_coeffs
        M2 = self.device_2.camera_matrix
        d2 = self.device_2.distortion_coeffs
        self.rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(object_points,
                                                               self.stereo_image_points_1,
                                                               self.stereo_image_points_2,
                                                               M1, d1, M2, d2,
                                                               self.image_size,
                                                               flags=self.calibration_flags)
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(M1, d1, M2, d2, self.image_size, R, T, None, None, flags=0,
                                                         alpha=-1)

        self.unrectified_to_rectified_transform = np.copy(R1)
        self.rectified_to_unrectified_transform = np.linalg.inv(R1)

        # manually set c_x and c_y to be equal to the principal point of left camera in reprojection matrix (so that the 3D coordinate of the centre of the depth image corresponds to x=0,y=0)
        self.Q[0][3] = -M1[0][2]  # c_x
        self.Q[1][3] = -M1[1][2]  # c_y
        self.map_x1, self.map_y1 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, self.image_size, cv2.CV_32F)
        self.map_x2, self.map_y2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, self.image_size, cv2.CV_32F)

        """
        T is given as a (3,1) matrix instead of a (1,3)- reshape it for ease.
        It is also given as the translation from the calibration to the origin instead of vice versa- times by -1 to get position relative to origin.
        Similarly, R defines the rotation matrix to get from camera direction to origin direction (z axis). Invert to get the opposite
        """
        T = T.reshape(3) * -1
        R = np.linalg.inv(R)
        self.R_1 = R_1
        self.T_1 = T_1
        self.T_2 = self.T_1 + T
        self.R_2 = np.matmul(self.R_1, R)
        self.T = T
        self.R = R
        self.device_1.set_position_and_orientation(self.R_1, self.T_1)
        self.device_2.set_position_and_orientation(self.R_2, self.T_2)
        return self.rms

    def get_images(self):
        return self.device_1.img, self.device_2.img

    def get_rectified_images(self):
        if self.device_1.img is None or self.device_2.img is None:
            return None, None

        img_1 = cv2.remap(self.device_1.img, self.map_x1, self.map_y1, cv2.INTER_LINEAR)
        img_2 = cv2.remap(self.device_2.img, self.map_x2, self.map_y2, cv2.INTER_LINEAR)
        return img_1, img_2

    def get_undistorted_images(self):
        if self.device_1.img is None or self.device_2.img is None:
            return None, None
        img_1 = self.device_1.undistort_image(self.device_1.img)
        img_2 = self.device_2.undistort_image(self.device_2.img)
        return img_1, img_2

    def scale_reprojection_matrix(self, scale_factor):
        Q = np.copy(self.Q)
        Q[:, 3] *= scale_factor
        return Q

    def depth_to_disparity(self, depth):
        # d = cx - cx' - fx Tx/Z' = A + B
        # A = cx-cx' = (cx-cx')/Tx * Tx = Q[3][3]/-Q[3][2]
        # B = -fx Tx/Z'= Q[2][3]/Q[3][2]/depth

        A = self.Q[3][3] / -self.Q[3][2]

        B = self.Q[2][3] / self.Q[3][2] / depth
        return (A + B)

    def disparity_to_depth(self, disparity):
        # Z' = Z/W = fTx / (cx - cx' -d)
        # Z = fx = Q[2][3]
        # W = (cx-cx')/Tx + -d/Tx
        # -1/Tx =Q[3][2]; (cx-cx)'/Tx = Q[3][3]
        W = self.Q[3][3] + disparity * self.Q[3][2]
        Z = self.Q[2][3]
        return Z / W

    def disparity_to_position(self, u, v, d):
        """
        Q =
                1       0        0       -c_x
                0       1        0       -c_y
                0       0        1       fx
                0       0       -1/T_x   (c_x - c_x')/T_x
        X = u - cx = u + Q[0][3]
        Y = v - cy = v + Q[1][3]
        Z = fx = Q[2][3]
        W = - d/Tx + (cx-cx')/Tx  = d*Q.at(3,2) + Q.at(3,3)
        posn = (X/W,Y/W,Z/W)
        """

        X = u + self.Q[0][3]
        Y = v + self.Q[1][3]
        Z = self.Q[2][3]
        W = d * self.Q[3][2] + self.Q[3][3]
        return np.array([X / W, Y / W, Z / W])

    def create_depth_map(self, mask=None, min_distance=0.6, max_distance=1.4,
                         block_size=7, speckle_window_size=31, apply_filter=False):
        img_1, img_2 = self.get_rectified_images()

        if mask is None:
            mask = np.ones((img_1.shape[0], img_1.shape[1]))

        disp1 = int(self.depth_to_disparity(min_distance))
        disp2 = int(self.depth_to_disparity(max_distance))
        min_disp = min(disp1, disp2) - min(disp1, disp2) % 16;
        max_disp = max(disp1, disp2) + (16 - max(disp1, disp2) % 16);

        num_disp = max_disp - min_disp
        if len(img_1.shape) == 2:
            nChannels = 1
        else:
            nChannels = img_1.shape[2]

        sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp,
                                     blockSize=block_size)
        sgbm.setP1(2 * nChannels * block_size ** 2)
        sgbm.setP2(4 * nChannels * block_size ** 2)
        sgbm.setSpeckleWindowSize(speckle_window_size)
        sgbm.setSpeckleRange(3)

        if apply_filter:
            sigma = 0.6
            lmbda = 16000.0
            left_matcher = sgbm
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher);
            left_disp = left_matcher.compute(img_1, img_2)
            right_disp = right_matcher.compute(img_2, img_1)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            left_disp_filtered = wls_filter.filter(left_disp, img_1, disparity_map_right=right_disp);
            img_disp = left_disp_filtered.astype(np.float32) / 16.0
        else:
            img_disp = sgbm.compute(img_1, img_2).astype(np.float32) / 16.0

        # img_disp = cv2.blur(img_disp,(15,15))

        img_disp[mask < 1] = min_disp
        img_depth = cv2.reprojectImageTo3D(img_disp, self.Q)

        for x in range(img_depth.shape[1]):
            for y in range(img_depth.shape[0]):
                img_depth[y][x] = np.matmul(self.unrectified_to_rectified_transform, img_depth[y][x])

        return img_depth

    def get_focal_length(self):
        return self.Q[2][3]


def register_image(img_1_depth, img_2, device_2, img_1_mask=None, img_2_mask=None):
    height_dst, width_dst = img_1_depth.shape[:2]
    img_dst = np.zeros((height_dst, width_dst), np.uint8)
    height_src, width_src = img_2.shape[:2]

    img_fn = core.create_image_interpolation_fn(img_2)

    if img_1_mask is None:
        img_1_mask = np.ones((img_1_depth.shape[:2]))
    if img_2_mask is None:
        img_2_mask = np.ones((img_2.shape[:2]))

    for x in range(width_dst):
        for y in range(height_dst):
            u, v = device_2.get_pixel_coordinates_of_point(img_1_depth[y][x])
            if img_1_mask[y][x] == 0:
                continue
            if int(u) < 0 or int(u) >= width_src or int(v) < 0 or int(v) >= height_src:
                continue
            if img_2_mask[int(v)][int(u)] == 0:
                continue
            L = img_fn(u, v)
            img_dst[y][x] = int(L)
    return img_dst


def register_image_2(img_1_depth, img_2, device_2, img_1_mask=None, img_2_mask=None):
    height_dst, width_dst = img_1_depth.shape[:2]
    width_src, height_src = device_2.image_size

    if img_1_mask is None:
        img_1_mask = np.ones((height_dst, width_dst))
    if img_2_mask is None:
        img_2_mask = np.ones((height_src, width_src))

    src_points = []
    dst_points = []

    for x in range(width_dst):
        for y in range(height_dst):
            u, v = device_2.get_pixel_coordinates_of_point(img_1_depth[y][x])
            if img_1_mask[y][x] == 0:
                continue
            if int(u) < 0 or int(u) >= width_src or int(v) < 0 or int(v) >= height_src:
                continue
            if img_2_mask[int(v)][int(u)] == 0:
                continue
            src_points.append([u, v])
            dst_points.append([x, y])
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    img_warped = cv2.warpPerspective(img_2, M, (width_dst, height_dst))

    return img_warped
