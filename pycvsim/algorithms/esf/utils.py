from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import skimage


def edge_is_vertical(image: NDArray):
    """
    Determine if the edge is vertical or horizontal based on if there is more variation
    in the horizontal or vertical direction
    Based on http://burnsdigitalimaging.com/software/sfrmat/iso12233-sfrmat5/ rotatev2.m

    :return: True if vertical
    """

    nn = 3
    test_v = np.abs(np.mean(image[-nn, :]) - np.mean(image[nn, :]))
    test_h = np.abs(np.mean(image[:, -nn]) - np.mean(image[:, nn]))
    # visualise looking at a vertical edge- the difference between the first column
    # and the last column is going to be large. The difference between the first row
    # and the last row is going to be small. For a vertical edge, test_h > test_v
    return test_h > test_v


def get_edge_location_in_line_from_esf(data: NDArray, threshold: float, display: bool = False):
    """

    :param data:
    :param threshold:
    :param display:
    :return:
    """
    assert (len(data.shape) == 1)
    data = data.astype(np.float32)
    data_offset = data - threshold
    indices = np.arange(data.shape[0])
    spline = InterpolatedUnivariateSpline(indices, data_offset)
    roots = spline.roots()

    if len(roots) != 1:
        return None

    if display:
        plt.plot(indices, data)
        plt.axhline(y=threshold)
        for r in roots:
            plt.axvline(x=r)
        plt.show()

    return roots[0]


def get_edge_location_in_line_from_lsf(data: NDArray, display: bool = False):
    """
    Find the location of the edge in a line or row, using the location of the maxima in the lsf.
    This is done vaguely similar to the ISO 12233 approach-
    (1) Apply Hamming filter to row to reduce noise
    (2) Take derivating of data to get LSF
    (3) Get maxima of LSF (centroid of edge)
    :param data:
    :param display:
    :return:
    """
    assert (len(data.shape) == 1)
    data = data.astype(np.float32)

    indices = np.arange(data.shape[0])
    spline = InterpolatedUnivariateSpline(indices, data, k=4)
    roots = spline.derivative().roots()

    if len(roots) == 0:
        return

    cr_vals = spline(roots)
    edge_pos = roots[np.argmax(cr_vals)]

    if display:
        plt.plot(indices, data)
        plt.axhline(y=edge_pos)
        for r in roots:
            plt.axvline(x=r)
        plt.show()
    return edge_pos


def get_edge_equation(image: NDArray, order: int = 1, edge_detection_mode = "lsf"):
    """

    :param image:
    :param order:
    :param edge_detection_mode:
    :return:
    """
    image = image.astype(np.float32)
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)

    is_vertical = edge_is_vertical(image)

    if not is_vertical:
        # transpose image so that it is vertical
        image = np.transpose(image)
    threshold = None
    if edge_detection_mode == "esf":
        threshold = skimage.filters.thresholding.threshold_otsu(image)

    height, width = image.shape
    x_data, y_data = [], []
    for y in range(height):
        if edge_detection_mode == "lsf":
            x = get_edge_location_in_line_from_lsf(image[y])
        elif edge_detection_mode == "esf":
            x = get_edge_location_in_line_from_esf(image[y], threshold)
        else:
            raise Exception("Unknown edge detection mode {}".format(edge_detection_mode))
        if x is not None:
            x_data.append(x)
            y_data.append(y)

    if not is_vertical:
        # flip x and y now as we transposed earlier
        x_data, y_data = y_data, x_data

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_to_y = np.poly1d(np.polyfit(x_data, y_data, deg=order))
    y_to_x = np.poly1d(np.polyfit(y_data, x_data, deg=order))
    return x_data, y_data, x_to_y, y_to_x
