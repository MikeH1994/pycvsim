from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import skimage
import cv2
from pycvsim.core.image_utils import convert_to_8_bit
import scipy.special
import scipy.optimize
import scipy.signal

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


def get_edge_location_in_line_from_lsf(data_input: NDArray, display: bool = True, apply_hamming=True):
    """
    Find the location of the edge in a line or row, using the location of the maxima in the lsf.
    This is done vaguely similar to the ISO 12233 approach-
    (1) Apply Hamming filter to row to reduce noise
    (2) Take derivating of data to get LSF
    (3) Get maxima of LSF (centroid of edge)
    :param data_input :
    :param display:
    :return:
    """
    assert (len(data_input.shape) == 1)
    data_input = data_input.astype(np.float32)
    w = data_input.shape[0]

    if apply_hamming:
        window = scipy.signal.windows.hamming(w)
        data = data_input # data_input*window
    else:
        data = data_input

    indices = np.arange(w)
    esf = InterpolatedUnivariateSpline(indices, data, k=4)
    lsf = InterpolatedUnivariateSpline(indices, esf.derivative()(indices), k=4)
    roots = lsf.derivative().roots()

    if len(roots) == 0:
        return None

    cr_vals = lsf(roots)
    edge_pos = roots[np.argmax(np.abs(cr_vals))]

    if display:
        x_fit = np.linspace(0, w-1, int(w*10))

        plt.plot(x_fit, esf(x_fit), label="Cubic spline")
        plt.scatter(indices, esf(indices), label="Measured points")
        plt.legend(loc=0)
        plt.xlabel("Pixel index")
        plt.ylabel("Intensity")
        plt.title("Edge profile for column")

        plt.figure()
        plt.plot(x_fit, lsf(x_fit), label="Cubic spline")
        plt.scatter(indices, lsf(indices), label="Measured points")
        plt.axvline(x=edge_pos, linestyle="--", label="Calculated edge position")
        plt.legend(loc=0)
        plt.xlabel("Pixel index")
        plt.ylabel("Intensity")
        plt.title("Line spread function for column")
        plt.show()
    return edge_pos


def get_edge_from_image(image_roi: NDArray, edge_detection_mode: str = "lsf", display=False):
    """

    :param image_roi:
    :param edge_detection_mode:
    :return:
    """
    image_roi = image_roi.astype(np.float32)
    if len(image_roi.shape) == 3:
        image_roi = np.mean(image_roi, axis=-1)

    is_vertical = edge_is_vertical(image_roi)
    if is_vertical:
        # transpose image so that it is horizontal (so we can avoid any potential
        # infinite gradients)
        image_roi = np.transpose(image_roi)
    threshold = None
    if edge_detection_mode == "esf":
        threshold = skimage.filters.thresholding.threshold_otsu(image_roi)

    height, width = image_roi.shape
    x_data, y_data = [], []
    for x in range(width):
        display_col = display and x == 0
        if edge_detection_mode == "lsf":
            y = get_edge_location_in_line_from_lsf(image_roi[:, x], display=display_col)
        elif edge_detection_mode == "esf":
            y = get_edge_location_in_line_from_esf(image_roi[:, x], threshold, display=display_col)
        else:
            raise Exception("Unknown edge detection mode {}".format(edge_detection_mode))
        if y is not None:
            x_data.append(x)
            y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    params = np.polyfit(x_data, y_data, deg=1)
    fit_fn = np.poly1d(params)
    x0, x1 = 0, width - 1.0
    y0, y1 = fit_fn(x0), fit_fn(x1)

    if is_vertical:
        # if we transposed it earlier, re-transpose to get back to original
        # orientation
        x_data, y_data = y_data, x_data
        x0, y0, x1, y1 = y0, x0, y1, x1

    if display:
        if is_vertical:
            image_roi = np.transpose(image_roi)
        img8 = convert_to_8_bit(image_roi, return_as_rgb=True)
        plt.figure()
        plt.imshow(img8)
        plt.scatter(x_data, y_data, s=10, color='r')
        plt.show()

    return np.array([x0, y0]), np.array([x1, y1]), x_data, y_data


def bin_data(esf_x: NDArray, esf_f: NDArray, bins_per_pixel: int = 4):
    bin_width = 1.0 / bins_per_pixel

    x0 = np.round(np.min(esf_x) * bins_per_pixel) / float(bins_per_pixel) - bin_width
    x1 = np.round(np.max(esf_x) * bins_per_pixel) / float(bins_per_pixel) + bin_width
    n_bins = int((x1 - x0) * bins_per_pixel) + 1

    bin_centres = []
    bin_values = []
    bin_std = []
    bin_range = []

    for x in np.linspace(x0, x1, n_bins):
        bin_lower = x - bin_width / 2.0
        bin_upper = x + bin_width / 2.0
        indices = np.where((esf_x >= bin_lower) & (esf_x < bin_upper))
        if np.count_nonzero(indices) == 0:
            continue
        bin_centres.append(x)
        bin_values.append(np.mean(esf_f[indices]))
        std = 0.0 if np.count_nonzero(indices) < 2 else np.std(esf_f[indices])
        half_range = 0.0 if np.count_nonzero(indices) < 2 else (np.max(esf_f[indices]) - np.min(esf_f[indices])) / 2.0
        bin_std.append(std)
        bin_range.append(half_range)
    bin_centres = np.array(bin_centres)
    bin_values = np.array(bin_values)
    bin_std = np.array(bin_std)
    bin_range = np.array(bin_range)
    return bin_centres, bin_values, bin_std, bin_range


def normalise_data(esf_x: NDArray, esf_f: NDArray, display=False, mode="fit"):
    def fn(x, x0_, y0_, a_, b_):
        f = y0_ + a_ * scipy.special.expit((x - x0_) / b_)
        return f

    # if the left side of the plot is greater than the right side, invert it
    if np.mean(esf_f[esf_x < 0]) > np.mean(esf_f[esf_x > 0]):
        esf_x *= -1.0

    if mode == "fit":
        p0 = [0, np.min(esf_f), np.max(esf_f) - np.min(esf_f), 1]
        [x0, y0, a, _], _ = scipy.optimize.curve_fit(fn, esf_x, esf_f, p0=p0)
        lower_val = y0
        upper_val = y0 + a
        x_offset = x0
    elif mode == "bin":
        bin_centres, bin_values, _, _ = bin_data(esf_x, esf_f, bins_per_pixel=1)
        lower_val = np.min(bin_values)
        upper_val = np.max(bin_values)
        offset_data = (esf_f - lower_val) / (upper_val - lower_val) - 0.5
        interpolation_fn = scipy.interpolate.InterpolatedUnivariateSpline(bin_centres, offset_data, k=2, ext=1)
        where_y_is_zero_point_five = interpolation_fn.roots()
        if len(where_y_is_zero_point_five) == 1:
            x_offset = where_y_is_zero_point_five[0]
        else:
            x_offset = 0
    else:
        raise Exception("Unknown mode")

    esf_f -= lower_val
    esf_f /= (upper_val - lower_val)
    esf_x -= x_offset

    return esf_x, esf_f
