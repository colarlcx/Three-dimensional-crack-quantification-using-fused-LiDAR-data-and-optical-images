# -*- coding: utf-8 -*-
import numpy as np
from resample import pyramid_down
from resample import pyramid_up


def image_to_gaussian_pyramid(image, level, cut_size=(3, 3)):
    """
    Build gaussian pyramid for an image. The size of the output component is
    arranged in descending order.

    Parameters
    ----------
    image: input image data read by opencv.
    level: level of output pyramid.
    cut_size: the minimal size of pyramid component, smaller than which the
        building process will be stopped.
    """
    gaussian_pyramid = [image]
    if level <= 1:
        return gaussian_pyramid

    for i in range(level - 1):
        # check down-sampled image size, should be >= cut_size
        height, width = image.shape[:2]
        height_down = (height + 1) // 2
        width_down = (width + 1) // 2
        if width_down < cut_size[0] or height_down < cut_size[1]:
            break
        # down sample
        image = pyramid_down(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid


def gaussian_to_laplacian_pyramid(gaussian_pyramid):
    """
    Build a laplacian pyramid from gaussian pyramid. The size of the output
    component is arranged in ascending order.
    """
    laplacian_pyramid = [gaussian_pyramid[-1]]
    level = len(gaussian_pyramid)
    if level == 1:
        return laplacian_pyramid

    for i in range(level - 1, 0, -1):
        up_size = gaussian_pyramid[i - 1].shape[:2][::-1]
        image_up = pyramid_up(gaussian_pyramid[i], up_size)
        # compute difference, use float type to avoid exceeding uint8 limit
        diff = np.float32(gaussian_pyramid[i - 1]) - np.float32(image_up)
        laplacian_pyramid.append(diff)
    return laplacian_pyramid


def image_to_laplacian_pyramid(image, level, cut_size=(3, 3)):
    """
    Build a laplacian pyramid from an image. The size of the output component
    is arranged in an ascending order.

    Parameters
    ----------
    image: input image data read by opencv.
    level: level of output pyramid.
    cut_size: the minimal size of pyramid component, smaller than which the
        building process will be stopped.
    """
    gaussian_pyramid = image_to_gaussian_pyramid(image, level, cut_size)
    laplacian_pyramid = gaussian_to_laplacian_pyramid(gaussian_pyramid)
    return laplacian_pyramid


def laplacian_pyramid_to_image(laplacian_pyramid):
    """
    Reconstruct an image from laplacian pyramid.
    """
    image = laplacian_pyramid[0]
    level = len(laplacian_pyramid)
    for i in range(1, level):
        up_size = laplacian_pyramid[i].shape[:2][::-1]
        image = pyramid_up(image, up_size, np.float32)
        image = np.float32(image) + laplacian_pyramid[i]
    image = np.uint8(np.clip(np.round(image), 0, 255))
    return image


def get_pyramid_index(original_index, level):
    """
    Get the index of a certain pyramid component corresponding to an index of
    original image

    Parameters
    ----------
    original_index: the index of original image.
    level: level for pyramid component.
    """
    if level < 0:
        raise ValueError("level can NOT be less than 0")

    if level == 0:
        return original_index

    base = 2 ** level
    mod = original_index % base

    if base == 2 * mod:
        # decimal part is 0.5
        return int(round(original_index / base / 2)) * 2
    else:
        return int(round(original_index / base))

