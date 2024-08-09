from typing import Tuple
from typing import Union, List
import cv2
import numpy as np
from numpy.typing import NDArray
from .colour import get_colour


def overlay_points_on_image(image: NDArray, keypoints_list: Union[List[NDArray], NDArray],
                            radius=3, dx=5, label=True, color=None) -> NDArray:
    """

    :param image:
    :param keypoints_list:
    :param radius:
    :param dx:
    :param label:
    :param color:
    :return:
    """
    assert(len(image.shape) == 3 and image.shape[2] == 3), "Image suppled should be rgb- shape: {}".format(image.shape)
    image = np.copy(image)
    image = np.ascontiguousarray(image) # I don't know why but cv2.rectangle fails otherwise
    keypoints_list = np.array(keypoints_list)
    if len(keypoints_list.shape) == 2:
        keypoints_list = keypoints_list.reshape((1, keypoints_list.shape[0], keypoints_list.shape[1]))
    assert(len(keypoints_list.shape) == 3)
    assert(keypoints_list.shape[-1] == 2 or keypoints_list.shape[-1] == 3)

    # draw keypoints
    image_points = np.zeros(image.shape, dtype=np.uint8)
    n_objects = len(keypoints_list)
    for i in range(n_objects):
        keypoints = keypoints_list[i]
        n_points = keypoints.shape[0]
        for n in range(n_points):
            point = keypoints[n]
            centre = (int(point[0]), int(point[1]))

            point_color = get_colour(i + 1) if color is None else color

            image_points = cv2.circle(image_points, centre, radius, point_color, -1)
            if label:
                cv2.putText(image_points, '{}'.format(n+1), org=(centre[0] + dx, centre[1] + dx), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=color, thickness=1, lineType=2)
    if image.dtype == np.float32:
        image_points = image_points.astype(np.float32)/255.0
    image_points_intensity = np.zeros(image_points.shape)
    image_points_intensity[:, :, :] = np.sum(image_points, axis=-1).reshape(*image_points.shape[:2], 1)
    image[image_points_intensity > 0] = image_points[image_points_intensity > 0]
    return image


def pad_image(img: NDArray, dst_size: Tuple[int, int]):
    """

    :param img:
    :param dst_size:
    :return:
    """
    dst_width, dst_height = dst_size
    assert (img.shape[0] <= dst_height and img.shape[1] <= dst_width)
    top = (dst_height - img.shape[0]) // 2
    bottom = dst_height - img.shape[0] - top
    left = (dst_width - img.shape[1]) // 2
    right = dst_width - img.shape[1] - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    assert (img.shape[0] == dst_height and img.shape[1] == dst_width)
    return img


def crop_image(img: NDArray, dst_size: Tuple[int, int]):
    dst_width, dst_height = dst_size
    assert (img.shape[0] >= dst_height and img.shape[1] >= dst_width)

    y0 = (img.shape[0] - dst_height) // 2
    x0 = (img.shape[1] - dst_width) // 2

    img = np.copy(img[y0:y0+dst_height, x0:x0+dst_width])
    assert (img.shape[0] == dst_height and img.shape[1] == dst_width)
    return img


def resize_image(img: NDArray, dst_size: Tuple[int, int],
                 exact_interpolation: bool = False, mode: str = 'pad'):
    """
    Resize an image_safe_zone to the given size, maintaining aspect ration by either cropping or padding the image_safe_zone based
    on the supplied arguments
    :param img:
    :param dst_size:
    :param exact_interpolation: if true (e.g if the exact value is important, such as in masks),
                                then use cv2.INTER_NEAREST
    :param mode: if 'pad', add a black border to maintain aspect ratio when resizing. If 'crop', trim the edges to
                 maintain aspect ratio
    :return:
    """
    assert(mode == 'pad' or mode == 'crop')
    src_height, src_width = img.shape[:2]
    dst_width, dst_height = dst_size
    k_x = dst_width / src_width
    k_y = dst_height / src_height
    interp_mode = cv2.INTER_NEAREST if exact_interpolation else cv2.INTER_CUBIC

    if mode == 'pad':
        if k_x < k_y:  # if the image_safe_zone needs to be resized more in the y direction
            # scale image_safe_zone so that width = dst_width and height < dst_height: then pad in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        else:  # if the image_safe_zone needs to be resized more in the x direction
            # scale image_safe_zone so that height = dst_height and width < dst_width: then pad in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        img = pad_image(img, dst_size)
        return img
    elif mode == 'crop':
        if k_x < k_y:  # if the image_safe_zone needs to be resized more in the y direction
            # scale image_safe_zone so that height = dst_height and width > dst_width: then crop in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        else:  # if the image_safe_zone needs to be resized more in the x direction
            # scale image_safe_zone so that width = dst_width and height > dst_height: then crop in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        img = crop_image(img, dst_size)
        return img
    else:
        raise Exception("Unknown mode -", mode)

def convert_to_8_bit(src: NDArray, min_val=None, max_val=None, return_as_rgb=False):
    # if a single image is supplied
    if src.dtype == np.uint8:
        if return_as_rgb and len(src.shape) == 1:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        return src

    min_val = np.min(src) if min_val is None else min_val
    max_val = np.max(src) if max_val is None else max_val

    if min_val > max_val:
        raise Exception("Invalid min and max bounds found!")

    if min_val == max_val:
        scale_factor = 1.0
    else:
        scale_factor = 255.0 / (max_val - min_val)
    img = src.astype(np.float32)
    img[img < min_val] = min_val
    img[img > max_val] = max_val
    img -= min_val
    img *= scale_factor
    img = img.astype(np.uint8)
    if return_as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def normalise_image(src, dark_img=None, img_min=None, img_max = None):
    dst = np.copy(src)
    if dark_img is None:
        dark_img = np.zeros(src.shape)
    dst -= dark_img
    if img_min is None:
        mid_val = (np.min(dst)+np.max(dst))/2.0
        img_min = np.mean(dst[dst < 0.7 * mid_val])
    if img_max is None:
        mid_val = (np.min(dst)+np.max(dst))/2.0
        img_max = np.mean(dst[dst > 1.3 * mid_val])
    dst -= img_min
    dst /= (img_max - img_min)
    return dst