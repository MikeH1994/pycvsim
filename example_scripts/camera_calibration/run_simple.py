from pycvsim.routines.calibration.imagesetgenerator import ImageSetGenerator
from pycvsim.camera.camera import Camera
from pycvsim.targets.checkerboardtarget import CheckerbordTarget
import numpy as np
import cv2


def create_checkerboard_points(board_size, dx):
    board_size = board_size
    dx = dx
    object_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
    object_points *= dx
    return object_points


def run(board_size=(7, 6), resolution=(800, 800), dx=0.1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera = Camera(pos=np.array([-0.0, 0.0, -1.5]), res=resolution, hfov=30.0)
    cameras = [camera]
    calibration_target = CheckerbordTarget(board_size, (dx, dx), board_thickness=0.02, color_bkg=(128, 0, 0))
    manager = ImageSetGenerator(cameras=cameras, calibration_target=calibration_target, n_horizontal=4, n_vertical=4,
                                n_angles=3, target_fill=0.4, max_alpha=0.0, max_beta=0.0)
    setpoints = manager.generate_setpoints()
    images = manager.run(setpoints)
    print("Setpoints created")
    # create_checkerboard_points(board_size, dx)  #
    objp = calibration_target.get_object_points(transformed=False).astype(np.float32)
    # Arrays to store object points and image points from all the unit_test_images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    print("{} unit_test_images".format(len(images)))
    for images_in_setpoint in images:
        image = images_in_setpoint[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(image, board_size, corners2, ret)
        cv2.imshow('img', image)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

    calibration_flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                              None, None,
                                                                              flags=calibration_flags)
    fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(camera_matrix,
                                                                                           gray.shape[::-1],  0.0, 0.0)

    print("Calculated camera matrix")
    print(camera_matrix)
    print("Actual camera matrix")
    print(camera.camera_matrix)
    print("Calculated Distortion coefficients")
    print(distortion_coeffs)
    print("Actual Distortion Coefficients")
    print(camera.distortion_coeffs)
    print("Fov")
    print(fovx, fovy)


if __name__ == "__main__":
    run()
