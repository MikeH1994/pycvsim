import psutil
import numpy as np
from scipy.interpolate import interp1d


def get_suggested_array_size(dtype=np.float32):
    available_memory = psutil.virtual_memory().available


def rescale_image(image, returned_dtype=np.uint8, kind="rescale"):
    if np.dtype(returned_dtype).kind == 'f':
        return image.astype(returned_dtype)

    min_value = np.iinfo(returned_dtype).min
    max_value = np.iinfo(returned_dtype).max
    image = image.astype(np.float32)
    if kind == "rescale":
        image = min_value + (max_value-min_value)*(image - np.min(image)) / (np.max(image) - np.min(image))
    elif kind == "clip":
        image[image < min_value] = min_value
        image[image > max_value] = max_value
    elif kind == "clip_lower":
        image[image < min_value] = min_value
        if np.max(image) > max_value:
            image = min_value + (max_value-min_value)*(image - np.min(image)) / (np.max(image) - np.min(image))
    return image.astype(returned_dtype)


def clip_line_to_image(p0, p1, image_res):
    xres, yres = image_res
    if p0[0] == p1[0]:
        # edge is basically vertical, avoid divide by zero error
        if p0[1] > p1[1]:
            p0, p1 = p1, p0
        if p0[1] < 0:
            p0[1] = 0
        if p1[1] > yres - 1:
            p1[1] = yres - 1
    else:
        m = (p1[1] - p1[0]) / (p1[0] - p0[0])
        if gradient_is_vertical(m):
            if p0[1] > p1[1]:
                p0, p1 = p1, p0
            # create line x = f(y)
            interp_fn = interp1d(np.array([p0[1], p1[1]]),
                                 np.array([p0[0], p1[0]], dtype=np.float32),
                                 fill_value="extrapolate")

            # check if first point has y < 0
            if p0[1] < 0:
                p0 = np.array([interp_fn(0).item(), 0], dtype=np.float32)
            if p1[1] > yres - 1:
                p1 = np.array([interp_fn(yres - 1).item(), yres - 1], dtype=np.float32)
        else:
            # create line y = f(x)
            if p0[0] > p1[0]:
                p0, p1 = p1, p0
            interp_fn = interp1d(np.array([p0[0], p1[0]], dtype=np.float32),
                                 np.array([p0[1], p1[1]], dtype=np.float32),
                                 fill_value="extrapolate")

            if p0[0] < 0:
                p0 = np.array([0, interp_fn(0).item()], dtype=np.float32)
            if p1[0] > xres - 1:
                p1 = np.array([xres - 1, interp_fn(xres - 1).item()], dtype=np.float32)

    return p0, p1


def gradient_to_angle(m: float):
    return np.degrees(np.arctan(m))


def gradient_is_vertical(m: float):
    return np.abs(m) > 1
