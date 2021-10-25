import numpy as np


def image_with_color(color=(255, 255, 255), size=(8192, 8192)):
    color_image = np.ones(shape=[size[1], size[0], 3], dtype=np.uint8)
    for color_index, color_value in enumerate(color):
        color_image[:, :, color_index] = color_value
    return color_image


def overlay(front_img, back_img, position=(0, 0)):
    x_offset, y_offset = int(front_img.shape[0]),int(front_img.shape[1])

    back_img[y_offset:y_offset + front_img.shape[0], x_offset:x_offset + front_img.shape[1]] = front_img

    return back_img
