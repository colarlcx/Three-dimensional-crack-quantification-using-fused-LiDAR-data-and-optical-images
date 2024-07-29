# -*- coding: utf-8 -*-
import cv2
import numpy as np


def blur(image, kernel_scale=1.0):
    """
    Blur image using a fixed kernel. Kernel scale can be set.

    Parameters
    ----------
    image: image data read by opencv.
    kernel_scale: the scale factor of kernel.
    """
    blur_kernel = np.array(
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]]) / 256.
    blurred_image = cv2.filter2D(image, ddepth=-1,
                                 kernel=blur_kernel * kernel_scale,
                                 borderType=cv2.BORDER_REFLECT101)
    return blurred_image


def pyramid_down(image):
    """
    Down sample an image by 2x.

    Parameters
    ----------
    image: image data read by opencv.
    """
    blurred_image = blur(image)
    image_down = blurred_image[::2, ::2]
    return image_down


def pyramid_up(image, dst_size=None, dtype=np.uint8):
    """
    Up sample an image by 2x. The output size and data type can be set.

    Parameters
    ----------
    image: image data read by opencv.
    dst_size: the output size. Note that the difference of dst_size and
        2*image_size should be <=2.
    dtype: the output data type.
    """
    # check dst_size
    height, width = image.shape[:2]
    if dst_size is None:
        dst_size = (width * 2, height * 2)
    else:
        if abs(dst_size[0] - width * 2) > 2 or \
                abs(dst_size[1] - height * 2) > 2:
            raise ValueError(r'the difference of dst_size and 2*image_size '
                             r'should be <=2.')

    # create a new buffer that has the dst_size
    dst_width, dst_height = dst_size
    if image.ndim == 2:
        image_up = np.zeros(shape=(dst_height, dst_width), dtype=dtype)
    else:
        channel = image.shape[2]

        image_up = np.zeros(shape=(dst_height, dst_width, channel),
                            dtype=dtype)

    image_up[::2, ::2] = image
    image_up = blur(image_up, 4.0)
    return image_up

