import numpy as np
from PIL import ImageOps


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    
    img2 = np.zeros((expected_size[0], expected_size[1], 3))
    img2[padding[1]:-padding[3],
               padding[0]:-padding[2]] = img
    return img2, padding


def depadding(img, padding):
    img = np.array(img)
    return img[padding[1]:-padding[3],
               padding[0]:-padding[2]]


