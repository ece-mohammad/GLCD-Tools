#!/usr/bin/env python3

# --------------------------------------------------------------------------------------------------------------------
"""
This script converts .jpg, .jpeg and .png images to bitmaps.
:requirements: Python3.7+, numpy & PIL
:usage: Usage: python image2bitmap.py /in/image/path /out/image/path
"""

# ---------------------------------------------------------------------------------------------------------------------

import sys
from io import StringIO
import numpy
from argparse import ArgumentParser
from PIL import Image
import pathlib

# ---------------------------------------------------------------------------------------------------------------------

__version__ = "1.2"


# ---------------------------------------------------------------------------------------------------------------------

def get_rgb(img):
    """
    Extract RGB channels from given image

    :param img: image object (JPEG, JPG, PNG, etc) that has RGB channels
    :type img:
    :return: RGB channels
    :rtype:
    """
    channels_num = len(img.mode)
    img_array = numpy.array(img)
    r_ch, g_ch, b_ch, *_ = numpy.split(img_array, channels_num, axis=2)
    return r_ch, g_ch, b_ch


def rgb_to_grey(r_ch, g_ch, b_ch) -> numpy.ndarray:
    """
    Mixes RGB channels into greyscale

    :param r_ch: Red channel data
    :type r_ch: numpy.ndarray
    :param g_ch: Green channel data
    :type g_ch: numpy.ndarray
    :param b_ch: Blue channel data
    :type b_ch: numpy.ndarray
    :return: Greyscale channel
    :rtype: numpy.ndarray
    """
    grey = (0.3 * r_ch + 0.59 * g_ch + 0.11 * b_ch)
    return grey


def grey_to_bitmap(grey, threshold=128) -> numpy.ndarray:
    """
    Converts a greyscale channel to bitmap

    :param grey:
    :type grey: numpy.ndarray
    :param threshold: Value that separates black/white pixels. Pixels with value >= threshold will be white.
    :type threshold: int
    :return: resulting bitmap
    :rtype: numpy.ndarray
    """
    bmp = (grey // threshold) * 255
    return bmp.astype(numpy.uint8)


def save_bitmap_bytes(bmp_bytes: numpy.ndarray, out_name) -> None:
    """
    Save bitmap bytes to given file

    :param bmp_bytes:
    :type bmp_bytes:
    :param out_name:
    :type out_name:
    :return: None
    :rtype: None
    """
    file_content: StringIO = StringIO()

    rows, cols = bmp_bytes.shape
    header = f"#define ROWS     {rows}\n" \
             f"#define COLUMNS  {cols}\n" \
             "\n" \
             "\n" \
             f"typedef unsigned char uint8_t;\n" \
             f"typedef unsigned short int uint16_t;\n" \
             f"\n" \
             f"uint16_t n_rows = ROWS;\n" \
             f"uint16_t n_cols = COLUMNS;\n" \
             f"\n" \
             f"uint8_t bmp_bytes [ROWS][COLUMNS] = {{\n"

    byte_list = (bmp_bytes // 255).tolist()

    file_content.write(header)

    for row in byte_list:
        file_content.write("\t")
        file_content.write(
            ", ".join([f"{b:#04x}" for b in row])
        )
        file_content.write(", \n")

    with open(out_name, "w", newline="") as fh:
        fh.write(file_content.getvalue().strip().strip(","))
        fh.write("\n};\n\n")

    file_content.close()


def main() -> int:
    arg_parser = ArgumentParser(
        description="Convert a given image (JPGJPEG//PNG) into a bitmap image, "
                    "or extract its bitmap equivalent bytes into a file."
                    "It first converts image's RGB channels to single greyscale channel"
                    "then applies a mask (threshold value) on the greyscale pixels."
                    "The resulting pixels will be either black or white.",
    )

    oper_group = arg_parser.add_argument_group("operations")  # supported operations command group
    opt_group = arg_parser.add_argument_group("output options")  # supported options command group

    # input image
    arg_parser.add_argument(
        'in_img',
        help="input image path."
    )

    # output file
    arg_parser.add_argument(
        '-o', '--out_file',
        help="output image/file path.",
        default="out"
    )

    # maximum dimension (for a square image)
    opt_group.add_argument(
        '-m', '--max_dim',
        help="maximum dimension of output image (width or height) in pixels. "
             "Note: this option is overridden by -s if both options are given.",
        type=int, default=0
    )

    # size (width x height)
    opt_group.add_argument(
        '-s', '--size',
        help="output image size. Note: this option overrides -m if both options are given.",
        nargs=2,
        type=int,
        default=0
    )

    # greyscale to monochrome conversion threshold
    opt_group.add_argument(
        '-t', '--threshold',
        help="threshold for converting RGB pixels to bitmap.",
        type=int,
        default=128
    )

    # convert to bitmap & save as hexdump file
    oper_group.add_argument(
        '-b', '--bytes',
        help="extract bitmap equivalent bytes and save to output file.",
        action="store_true",
        default=False
    )

    # convert to bitmap & save as bitmap image
    oper_group.add_argument(
        '-c', '--convert',
        help="convert image to bitmap and save to output file.",
        action="store_true",
        default=True
    )

    cmd_args = arg_parser.parse_args(sys.argv[1:])

    out_file: pathlib.Path = pathlib.Path(cmd_args.out_file).absolute()
    out_file_dir: pathlib.Path = out_file.parent
    out_file_name: str = out_file.name

    image_file: pathlib.Path = pathlib.Path(cmd_args.in_img)

    # check image file
    if not image_file.is_file():
        print(f"input file: {cmd_args.in_img} doesn't exist!")
        return -1

    if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        print(f"Unsupported format: {image_file.suffix}. Allowed image formats: JPG/JPEG, PNG")
        return -1

    # open image file
    with Image.open(image_file) as img_obj:

        # check that image has RGB channels
        if len(img_obj.mode) < 3:
            print(f"Image {image_file} doesn't have RGB channel. Convert it to another format.")
            return -1

        # resize image
        if cmd_args.size:
            img_obj = img_obj.resize((cmd_args.size[0], cmd_args.size[1]))

        # set max dimensions
        elif cmd_args.max_dim:
            scale = cmd_args.max_dim / max(img_obj.size)
            new_width = int(img_obj.width * scale)
            new_height = int(img_obj.height * scale)
            img_obj = img_obj.resize((new_width, new_height))

        # get image r, g, b channels
        r_ch, g_ch, b_ch = get_rgb(img_obj)

        # convert image to grey scale
        gs_img = rgb_to_grey(r_ch, g_ch, b_ch)

        # convert greyscale to bitmap
        bmp_bytes = grey_to_bitmap(gs_img, cmd_args.threshold)
        bmp_bytes = bmp_bytes.reshape(gs_img.shape[0], gs_img.shape[1])

        # save bitmap bytes to a hex dump file
        if cmd_args.bytes:
            save_bitmap_bytes(bmp_bytes, out_file.with_suffix(".h"))

        # save bitmap to file
        if cmd_args.convert:
            bmp_img = Image.fromarray(bmp_bytes, mode='L')
            bmp_img.save(out_file.with_suffix(".bmp"))

    return 0


if __name__ == '__main__':
    ret: int = main()
    sys.exit(ret)
