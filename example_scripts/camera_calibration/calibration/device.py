import cv2
import numpy as np
import os
import pickle
import copy
import math
import matplotlib.pyplot as plt
import scipy.optimize
from enum import Enum
from numpy.typing import NDArray
from pycvsim.core.pinhole_camera_maths import focal_length_to_fov



class Device:
    """
    A class that handles the reading of each image, finding points on a calibration target and then

    """

    def __init__(self, name, image_size):
        self.calibration_flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
        self.name = name
        self.image_points = []
        self.image_point_keys = []
        self.image_size = image_size

        self.rms = 0
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.R = None
        self.T = None
        self.axes = None

    def add_calibration_point(self, img: NDArray, key, board_size, use_larger_blobs=False, mode="checkerboard"):
        assert(mode == "checkerboard" or mode == "symmetric_grid" or mode == "asymmetric_grid")
        success = False
        img_overlayed = np.copy(img)
        if mode == "checkerboard":
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            success, image_points = cv2.findChessboardCorners(img_gray, board_size, None)
            if success:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                image_points = cv2.cornerSubPix(img_gray, image_points, (11, 11), (-1, -1), criteria)
        else:
            grid = cv2.CALIB_CB_SYMMETRIC_GRID if mode == "symmetric_grid" else cv2.CALIB_CB_ASYMMETRIC_GRID
            if use_larger_blobs:
                params = cv2.SimpleBlobDetector_Params()
                params.maxArea = 1e5
                blob_detector = cv2.SimpleBlobDetector_create(params)
                success, image_points = cv2.findCirclesGrid(img, board_size, grid, blobDetector=blob_detector)
            else:
                success, image_points = cv2.findCirclesGrid(img, board_size, grid)
            if not success and not use_larger_blobs:
                # as we didn't find it using the default arguments, try again with larger blob size
                return self.add_calibration_point(image_points, key, board_size, use_larger_blobs=True, mode=mode)
        if success:
            img_overlayed = cv2.drawChessboardCorners(img, board_size, image_points, True)
            self.image_points.append(image_points)
            self.image_point_keys.append(key)
        return success, img_overlayed

    def calibrate(self, object_points, alpha=None):
        object_points = [object_points.astype(np.float32) for _ in range(len(self.image_points))]
        # beware! opencv functions uses (x,y) coordinates instead of (y,x)
        self.rms, self.camera_matrix, self.distortion_coeffs, r, t = cv2.calibrateCamera(object_points,
                                                                                         self.image_points,
                                                                                         self.image_size, None, None,
                                                                                         flags=self.calibration_flags)
        if alpha is not None:
            newcamera_matrix, roi = cv2.getOptimalNewcamera_matrix(self.camera_matrix, self.distortion_coeffs,
                                                                   self.image_size, alpha)
            self.camera_matrix = newcamera_matrix
        return self.rms

    def undistort_image(self, img):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            raise Exception("In undistort_image(): camera calibration for {} is not computed")
        img_undistorted = cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)
        img_undistorted[img_undistorted == 0] = np.min(img_undistorted[img_undistorted > 0])
        return img_undistorted

    def clear_all(self):
        self.image_points = []
        self.image_point_keys = []
        self.rms = 0
        self.camera_matrix = None
        self.distortion_coeffs = None

    def set_position_and_orientation(self, R, T):
        self.R = R
        self.T = T
        x_axis = np.matmul(self.R, np.array([1, 0, 0]))
        y_axis = np.matmul(self.R, np.array([0, 1, 0]))
        z_axis = np.matmul(self.R, np.array([0, 0, 1]))
        self.axes = [x_axis, y_axis, z_axis]


def optimise_calibration_target_fn(P, devices, calibration_target, i):
    """
    Parameters
    ----------
    P
        The next guess for the x-y coordinates of point i on the calibration
        target. Should be a numpy array of length 2
    devices
        A list of camera_calibration.Device instance. The each calibration must already have
        gone through each calibration image and found the calibration pattern.
    calibration_target
        A CalibrationTarget instance. We will change coordinates of
        calibration_target.object_points[i] to see if we can improve the rms of the camera
        calibrations.
    i
        the index of the point on the calibration target we want to change.
    """
    calibration_target.object_points[i][:2] = P
    rms_values = [device.calibrate(calibration_target) for device in devices]
    rms = np.mean(np.array(rms_values))
    return rms


def optimise_calibration_target(folderpath, devices, calibration_target_init, i_min, i_max, dx=0.0004):
    """
    Due to tolerances in the manufacture of the calibration target, the actual
    centroids of the calibration points may vary somewhat compared to the predicted
    positions. The performance of the calibration can be improved by optimization
    approaches, varying the coordinates of each point slightly to try and improved
    results.


    Parameters
    ----------
    folderpath
        the folderpath of the calibration images that we are going to use in the
        camera calibrations
    devices
        a list of camera_calibration.Device instances
    calibration_target_init
        the default calibration target we want to optimise. In the example cases used,
        it should be a 7x4 grid with centroids 40mm apart, i.e.
        calibration_target_init = core.CalibrationTarget((7,4),0.04)
    i_min
        the index of the first image in the calibration set (0 in the example used)
    i_max
        the index of the last image in the calibration set (34 in the example used)
    dx
        the amount by which we will vary the calibration target positions in any direction.
        The stated tolerance in the manufacture of the calibration target was 0.2mm.
        Setting the default value to 0.4mm to give a bit of extra leeway.
    """
    print("Beginning calibration target optimisation")
    devices = copy.deepcopy(devices)
    calibration_target = copy.deepcopy(calibration_target_init)
    for device in devices:
        device.clear_all()
    for i in range(i_min, i_max + 1):
        for device in devices:
            device.open_img(folderpath, i, return_as_8_bit=True, invert_thermal=True)
            success = device.add_calibration_point(calibration_target_init, i)
    for device in devices:
        device.clear_image()

    rms = np.mean(np.array([device.calibrate(calibration_target) for device in devices]))
    print("rms before optimisation = {:.4f}".format(rms))

    object_points = calibration_target.object_points
    # for each point on the calibration target, optimise the x-y coordinates to improve
    # the mean rms of calibration camera calibrations
    for i in range(calibration_target.n_points):
        P = calibration_target.object_points[i][:2]
        bounds = [(P[0] - dx, P[0] + dx), (P[1] - dx, P[1] + dx)]

        result = scipy.optimize.minimize(optimise_calibration_target_fn, P,
                                         args=(devices, calibration_target, i),
                                         bounds=bounds)
        P_opt = result.x.astype(np.float32)
        calibration_target.object_points[i][:2] = P_opt
        rms = np.mean(np.array([device.calibrate(calibration_target) for device in devices]))
        print("{}/{} - rms = {:.4f}".format(i, calibration_target.n_points, rms))

    print("Calibration target optimisation complete")
    return calibration_target
