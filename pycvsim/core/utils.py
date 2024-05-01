import psutil
import numpy as np


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
