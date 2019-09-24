#!/usr/bin/env python3


import os
from PIL import Image
import numpy


def jpg_to_bitmap_numpy(img, threshold):
    """
    Convert an JPEG image to bitmap
    """
    # open image
    orig_img = Image.open(img)

    img_array = numpy.array(orig_img)

    # split rgb channels
    r_ch, g_ch, b_ch = numpy.split(img_array, 3, axis=2)

    # RGB to gray scale
    bitmap = numpy.array(
        [
            (0.3 * i[0] + 0.59 * i[1] + 0.11 * i[2]) for i in zip(r_ch, g_ch, b_ch)
        ]
    )

    # reshape new image
    bitmap = bitmap.reshape(img_array.shape[0], img_array.shape[1])

    # apply threshold on gray scale (gray scale to bitmap)
    bitmap = numpy.dot(
        (bitmap > threshold).astype(float),
        255
    )

    bmp_img = Image.fromarray(
        bitmap.astype(numpy.uint8)
    )

    # save output as new image
    return bmp_img


def png_to_bitmap_numpy(img, threshold):
    """
    Convert PNG image to bitmap
    """
    # open image
    orig_image = Image.open(img)

    n_channels = len(orig_image.split())

    # check # of channels
    # 4 channels (RGBA)
    if n_channels < 3:
        print("Can't convert given image: {}!"
              " convert it into a JPG/JPEG image first".format(
            img, n_channels
        )
        )
        return orig_image

    elif n_channels == 4:

        # get rgb channels
        img_array = numpy.array(orig_image)

        r_ch, g_ch, b_ch, _ = numpy.split(img_array, 4, axis=2)

    elif n_channels == 3:

        # get rgb channels
        img_array = numpy.array(orig_image)

        r_ch, g_ch, b_ch = numpy.split(img_array, 3, axis=2)

    # convert RGB > gray scale
    bitmap = numpy.array(
        [
            (0.3 * i[0] + 0.59 * i[1] + 0.11 * i[2]) for i in zip(r_ch, g_ch, b_ch)
        ]
    )

    # reshape new image 
    bitmap = bitmap.reshape(img_array.shape[0], img_array.shape[1])

    # apply threshold on gray scale (gray scale to bitmap)
    bitmap = numpy.dot(
        (bitmap > threshold).astype(float),
        255
    )

    bmp_img = Image.fromarray(
        bitmap.astype(numpy.uint8)
    )

    return bmp_img


if __name__ == '__main__':
    """   Test JPEG Conversion    """
    test_img = "test.jpg"
    new_img = jpg_to_bitmap_numpy(test_img, 128)
    new_img.save(test_img + ".bmp")

    """   Test PMG Conversion   """
    test_img = "test.png"
    new_img = png_to_bitmap_numpy(test_img, 128)
    new_img.save(test_img + ".bmp")

    """   Test PMG Conversion   """
    test_img = "test2.png"
    new_img = png_to_bitmap_numpy(test_img, 128)
    new_img.save(test_img + ".bmp")
