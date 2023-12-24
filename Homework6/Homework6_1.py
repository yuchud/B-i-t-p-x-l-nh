import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

HEIGHT, WIDTH = 256, 256
PLT_ROW, PLT_COL, plt_idx = 2, 4, 1
PLT_FONT_SIZE = 7
plt.rcParams.update({'font.size': PLT_FONT_SIZE})

KERNEL_SIZE = (3, 3)
KERNEL = np.ones(KERNEL_SIZE)

camera9 = np.fromfile(open('camera9.bin'), dtype=np.uint8).reshape(HEIGHT, WIDTH)
camera99 = np.fromfile(open('camera99.bin'), dtype=np.uint8).reshape(HEIGHT, WIDTH)


def full_scale_contrast(src):
    max_v = np.max(src)
    min_v = np.min(src)
    return np.round((src - min_v) / (max_v - min_v) * 255.0)


def showimg(src, title=None, cmap=None, is_histogram=False):
    global PLT_ROW, PLT_COL, plt_idx
    plt.subplot(PLT_ROW, PLT_COL, plt_idx)
    plt_idx += 1
    if is_histogram:
        plt.plot(src)
    else:
        plt.imshow(src, cmap)
        plt.axis(False)
    plt.title(title)


def median_filter(_image):
    image_height, image_width = _image.shape[0], _image.shape[1]
    kernel_height, kernel_width = KERNEL.shape[0], KERNEL.shape[1]  # kernel size m x n

    filtered_image = np.zeros((image_height, image_width))
    padded_image = np.zeros((image_height + kernel_height // 2 * 2, image_width + kernel_width // 2 * 2))
    padded_image[kernel_height // 2: kernel_height // 2 + image_height,
    kernel_width // 2: kernel_width // 2 + image_width] = _image

    for curr_height in range(image_height):
        for curr_width in range(image_width):
            filtered_image[curr_height][curr_width] = np.median(
                padded_image[curr_height:curr_height + kernel_height, curr_width:curr_width + kernel_width] * KERNEL)

    return filtered_image


def dilate(_image):
    image_height, image_width = _image.shape
    kernel_size = KERNEL_SIZE[0]
    res = deepcopy(_image)

    for current_height in range(image_height - kernel_size + 1):
        for current_width in range(image_width - kernel_size + 1):
            W = _image[current_height:current_height + kernel_size, current_width:current_width + kernel_size]
            res[current_height + kernel_size // 2][current_width + kernel_size // 2] = np.max(W * KERNEL)

    return res


def erode(_image):
    image_height, image_width = _image.shape
    kernel_size = KERNEL_SIZE[0]
    res = deepcopy(_image)

    for current_height in range(image_height - kernel_size + 1):
        for current_width in range(image_width - kernel_size + 1):
            W = _image[current_height:current_height + kernel_size, current_width:current_width + kernel_size]
            res[current_height + kernel_size // 2][current_width + kernel_size // 2] = np.min(W * KERNEL)

    return res


def morphological(_image, mode='opening'):
    if mode == 'opening':
        return dilate(erode(_image))

    if mode == 'closing':
        return erode(dilate(_image))

    return None


median_camera9 = median_filter(camera9)
opening_camera9 = morphological(camera9, 'opening')
closing_camera9 = morphological(camera9, 'closing')

median_camera99 = median_filter(camera9)
opening_camera99 = morphological(camera99, 'opening')
closing_camera99 = morphological(camera99, 'closing')

showimg(camera9, 'camera9.bin', 'gray')
showimg(median_camera9, 'camera9.bin Median Filter', 'gray')
showimg(opening_camera9, 'camera9.bin Morphological Opening', 'gray')
showimg(closing_camera9, 'camera9.bin Morphological Closing', 'gray')

showimg(camera99, 'camera99.bin', 'gray')
showimg(median_camera99, 'camera99.bin Median Filter', 'gray')
showimg(opening_camera99, 'camera99.bin Morphological Opening', 'gray')
showimg(closing_camera99, 'camera99.bin Morphological Closing', 'gray')

plt.show()
