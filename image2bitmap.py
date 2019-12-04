#!/usr/bin/env python3

# --------------------------------------------------------------------------------------------------------------------
"""
This script converts .jpg, .jpeg and .png images to bitmaps.
:requirements: Python3
:usage: Usage: python image2bitmap.py /in/image/path /out/image/path
"""

# ---------------------------------------------------------------------------------------------------------------------

import os
import sys
import numpy
from argparse import ArgumentParser
from PIL import Image

# ---------------------------------------------------------------------------------------------------------------------

__version__ = "1.2"

# ---------------------------------------------------------------------------------------------------------------------


def rgb_to_grey(r_ch, g_ch, b_ch):
    """
    Mixes RGB channels into greyscale
    :param r_ch: (numpy.ndarray) Red channel data
    :param g_ch: (numpy.ndarray) Green channel data
    :param b_ch: (numpy.ndarray) Blue channel data
    :return: (numpy.ndarray) Greyscale data
    """
    grey = (0.3 * r_ch + 0.59 * g_ch + 0.11 * b_ch)
    return grey


def grey_to_bitmap(grey, threshold=128):
    """
    Converts a greyscale channel to bitmap
    :param grey: (numpy.ndarray)
    :param threshold: Value that separates black/white pixels. Pixels with value >= threshold will be white.
    :return: (numpy.ndarray) resulting bitmap
    """
    bmp = (grey // threshold) * 255
    return bmp.astype(numpy.uint8)


def get_rgb(img):
    """
    :param img:
    :return:
    """
    channels_num = len(img.mode)
    img_array = numpy.array(img)
    r_ch, g_ch, b_ch, *_ = numpy.split(img_array, channels_num, axis=2)
    return r_ch, g_ch, b_ch


def save_bitmap_bytes(bmp_bytes, out_name):
    """
    :param bmp_bytes:
    :param out_name:
    :return:
    """
    rows, cols = bmp_bytes.shape
    header = "typedef unsigned char uint8_t;\n" \
             "typedef unsigned short int uint16_t;\n" \
             "\n" \
             "uint16_t n_rows = {rows};\n"\
             "uint16_t n_cols = {cols};\n"\
             "\n"\
             "uint8_t bmp_bytes [{rows}][{cols}] = {{\n".format(**{"rows": rows, "cols": cols})
    byte_list = bmp_bytes.tolist()
    with open(out_name, "w") as fh:
        fh.write(header)
        for row in range(rows):
            fh.write("\t"+", ".join([str(i) for i in byte_list[row]])+",\n")
        fh.write("}\n\n")


def main():
    arg_parser = ArgumentParser(
        description="Convert a given image (JPG/PNG) into a bitmap image, "
                    "or extract its bitmap equivalent bytes into a file."
                    "It first converts image's RGB channels to single greyscale channel"
                    "then applies a mask (threshold value) on the greyscale pixels."
                    "The resulting pixels will be either black or white.",
    )

    oper_group = arg_parser.add_argument_group("operations")
    opt_group = arg_parser.add_argument_group("output options")
    arg_parser.add_argument('in_img', help="input image path.")
    arg_parser.add_argument('-o', '--out_file', help="output image/file path.", default="out")
    opt_group.add_argument('-m', '--max_dim', help="maximum dimension of output image (width or height) in pixels. "
                                                   "Note: this option is overridden by -s if both options are given.",
                           type=int, default=0)
    opt_group.add_argument('-s', '--size', help="output image size. Note: this option overrides -m if both options are "
                                                "given.", nargs=2, type=int, default=0)
    opt_group.add_argument('-t', '--threshold', help="threshold for converting RGB pixels to bitmap.", type=int,
                           default=128)
    oper_group.add_argument('-b', '--bytes', help="extract bitmap equivalent bytes and save to output file.",
                            action="store_true", default=False)
    oper_group.add_argument('-c', '--convert', help="convert image to bitmap and save to output file.",
                            action="store_true", default=False)

    cmd_args = arg_parser.parse_args(sys.argv[1:])
    out_file = cmd_args.out_file
    out_file_dir = os.path.dirname(out_file)
    out_file_name, *_ = os.path.splitext(os.path.basename(out_file))

    if os.path.isfile(cmd_args.in_img):
        img_obj = Image.open(cmd_args.in_img)
        if img_obj.format.lower() in ('jpg', 'jpeg', 'png') and len(img_obj.mode) >= 3:
            if cmd_args.size:
                new_img = img_obj.resize((cmd_args.size[0], cmd_args.size[1]))
            elif cmd_args.max_dim:
                scale = cmd_args.max_dim / max(img_obj.size)
                new_width = int(img_obj.width*scale)
                new_height = int(img_obj.height*scale)
                new_img = img_obj.resize((new_width, new_height))
            else:
                new_img = img_obj

            r_ch, g_ch, b_ch = get_rgb(new_img)
            gs_img = rgb_to_grey(r_ch, g_ch, b_ch)
            bmp_bytes = grey_to_bitmap(gs_img, cmd_args.threshold)
            bmp_bytes = bmp_bytes.reshape(gs_img.shape[0], gs_img.shape[1])

            if cmd_args.bytes:
                save_bitmap_bytes(bmp_bytes, os.path.join(out_file_dir, out_file_name+".h"))

            if cmd_args.convert:
                bmp_img = Image.fromarray(bmp_bytes, mode='L')
                bmp_img.save(os.path.join(out_file_dir, out_file_name+".bmp"))

        else:   # unsupported format
            if img_obj.format.lower not in ("png", "jpg", "jpeg"):
                print("Unsupported format: {}. Allowed image formats: JPG/JPEG, PNG".format(img_obj.format))
            elif len(img_obj.mode) < 3:
                print("Image {} doesn't have RGB channel. Convert it to another format.".format(cmd_args.in_img))
            sys.exit(-1)
    else:   # in_img doesn't exist
        print("input file: {} doesn't exist!".format(cmd_args.in_img))
        sys.exit(-1)


if __name__ == '__main__':
    main()


