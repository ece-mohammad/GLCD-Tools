#!/usr/bin/env python3

# --------------------------------------------------------------------------------------------------------------------
"""
This script converts .jpg, .jpeg and .png images to bitmaps.
:requirements: Python3.7+, numpy & PIL
:usage: Usage: python image_converter.py /in/image/path /out/image/path
"""

# ---------------------------------------------------------------------------------------------------------------------

import pathlib
import sys
from argparse import ArgumentParser
from binascii import hexlify
from io import StringIO
from math import ceil
from typing import Optional, Tuple

import numpy
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------------------------------------------------

__version__ = "1.3"

# ---------------------------------------------------------------------------------------------------------------------
from more_itertools import chunked


class ImageConverter(object):

    def __init__(self, image_path: pathlib.Path):
        self.image_path: pathlib.Path = image_path
        self.image: Optional[Image.Image] = self.open_image(self.image_path)

    @staticmethod
    def check_image_file(image_file: pathlib.Path) -> bool:
        """
        check if image file exists
        :return: True if file exists, False otherwise
        :rtype: None
        """
        return image_file.exists() and image_file.is_file()

    @staticmethod
    def resize_image(image: Image.Image, width: int, height: int, padding: Optional[str] = None) -> Image.Image:
        """
        Resize given image to the given width and height. The image will be scaled to fill
        the given width and height. However, If a padding color was specified, the image is scaled
        to fill width and height without distortion, preserving the width/height ratio and the remaining pixels
        are filled with padding color.

        :param image: image to resize
        :type image: Image.Image
        :param width: new image width. If <= 0, original width is kept
        :type width: int
        :param height: new image height. If <= 0, original height is kept
        :type height: int
        :param padding: Padding color added around resized image
        :type padding: str
        :return: resized image
        :rtype: Image.Image
        """
        image_size: Tuple[int, int] = image.size
        if width <= 0:
            width = image_size[0]

        if height <= 0:
            height = image_size[1]

        if padding is None:
            return image.resize((width, height))
        else:
            return ImageOps.pad(image, (width, height), color=padding, centering=(0.5, 0.5))

    @staticmethod
    def scale_image(image: Image.Image, scale: float) -> Image.Image:
        """
        scale given image to the given scale. The image will be scaled

        :param image: image to scale
        :type image: Image.Image
        :param scale: scale applied to image dimensions, 0 < scale < 1 to scale down, 1 < scale < inf to scale up
        :type scale: float
        :return: scaled image
        :rtype: Image.Image
        """
        if scale <= 0:
            return image

        return ImageOps.scale(image, scale)

    @staticmethod
    def save_byte_array_to_file(byte_array: numpy.ndarray, fmt: str, fname: pathlib.Path) -> bool:
        """
        Save given byte array into a file of given format

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param byte_array: raw bytes as a numpy array
        :type byte_array: numpy.ndarray
        :param fmt: file format, must be one of the supported formats, otherwise the bytes are not saved
        :type fmt: str
        :param fname: file path
        :type fname: pathlib.Path
        :return: True if bytes were saved, False otherwise
        :rtype: bool
        """
        match fmt:
            # save as binary file
            case 'b':
                byte_array.tofile(fname.with_suffix(".bin"), sep="", format="%02x")

            # save as hexdump
            case 'x':
                with open(fname.with_suffix(".hex"), "wb") as bin_file:
                    for address, chunk in enumerate(chunked(byte_array.tobytes(order='C'), 16)):
                        bin_file.write(bytes(f"{address * 16:08X} ", "ascii"))
                        bin_file.write(hexlify(bytearray(chunk), ' ', 1))
                        bin_file.write(b'\n')

            # save as C file
            case 'c':
                rows, columns = byte_array.shape
                header: str = "#include <stdint.h>\n" \
                              "\n" \
                              "\n" \
                              "#define ROWS      {rows}\n" \
                              "#define COLUMNS   {columns}\n" \
                              "\n" \
                              "\n" \
                              "uint32_t {image_name}_rows = ROWS;\n" \
                              "uint32_t {image_name}_columns = COLUMNS;\n" \
                              "\n" \
                              "\n" \
                              "uint8_t {image_name}_bytes [ROWS][COLUMNS] = {{\n"

                footer: str = "};\n\n"

                with open(fname.with_suffix(".c"), "w") as c_file, StringIO() as temp_file:
                    for chunk in chunked(byte_array.tobytes(order='C'), columns):
                        temp_file.write(
                            f"    {{{', '.join([f'0x{i:02x}' for i in chunk])}}},\n"
                        )

                    c_file.write(
                        header.format(
                            rows=rows,
                            columns=columns,
                            image_name=fname.name
                        )
                    )
                    c_file.write(f"{temp_file.getvalue().rstrip().rstrip(',')}\n")
                    c_file.write(footer)

        return True

    @staticmethod
    def save_greyscale_image(gs_image: Image.Image, f_name: pathlib.Path) -> None:
        """
        Save given greyscale image

        :param gs_image: greyscale image
        :type gs_image: Image.Image
        :param f_name: greyscale image path
        :type f_name: pathlib.Path
        :return: None
        :rtype: None
        """
        gs_image.save(f_name.with_name(f"{f_name.name}_gs").with_suffix(".jpg"))

    @staticmethod
    def save_bitmap_image(bmp_image: Image.Image, f_name: pathlib.Path) -> None:
        """
        Save given bitmap image

        :param bmp_image: bitmap image
        :type bmp_image: Image.Image
        :param f_name: bitmap image path
        :type f_name: pathlib.Path
        :return: None
        :rtype: None
        """
        bmp_image.save(f_name.with_name(f"{f_name.name}_bmp").with_suffix(".bmp"))

    @staticmethod
    def rle_encode_byte_array(byte_array: numpy.ndarray) -> bytes:
        """
        Encode given byte array using RLE (Run Length Encoding)

        :param byte_array: byte array to encode
        :type byte_array: numpy.ndarray
        :return: encoded
        :rtype:
        """
        pass

    def rgb_to_grey(self, image: Image.Image) -> Image.Image:
        """
        Convert RGB image to greyscale

        :param image: Image object
        :type image: Image.Image
        :return: Greyscale image
        :rtype: Image.Image
        """
        return ImageOps.grayscale(image)

    def rgb_to_bitmap(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Converts RGB image to bitmap

        :param image: RGB image object
        :type image: Image.Image
        :param threshold: Value that separates black/white pixels. Pixels with value >= threshold will be white.
        :type threshold: int
        :return: Image.Image
        :rtype: Image.Image
        """
        grey: Image.Image = self.rgb_to_grey(image)
        bitmap_bytes: numpy.ndarray = numpy.array(grey)
        bitmap_bytes: numpy.ndarray = numpy.floor(bitmap_bytes / threshold) * 255
        bitmap_bytes = bitmap_bytes.astype('uint8')
        return Image.fromarray(bitmap_bytes, mode='L')

    def open_image(self, image_file: pathlib.Path) -> Optional[Image.Image]:
        """
        Open given image file, and return the opn image as PIL.Image object

        :param image_file: path to image file
        :type image_file: pathlib.Path
        :return: opened image file if it exists, otherwise return None
        :rtype: Image
        """
        image: Optional[Image.Image] = None
        if self.check_image_file(image_file):
            try:
                image = Image.open(image_file)
            except Exception as exc:
                pass
        return image

    def save_greyscale_bytes(self, gs_image: Image.Image, fmt: str, fname: pathlib.Path) -> bool:
        """
        Save given bytes to a file of given format

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param gs_image: greyscale image raw bytes
        :type gs_image: numpy.ndarray
        :param fmt: file format, must be one of the supported formats, otherwise the bytes are not saved
        :type fmt: str
        :param fname: file path
        :type fname: pathlib.Path
        :return: True if bytes were saved, False otherwise
        :rtype: bool
        """
        return self.save_byte_array_to_file(
            numpy.array(gs_image).astype('uint8'),
            fmt,
            fname.with_name(f"{fname.name}_gs")
        )

    def save_bitmap_bytes(self, bmp_image: Image.Image, fmt: str, fname: pathlib.Path, compression: int = 0) -> bool:
        """
        Save given bytes to a file of given format

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param bmp_image: bitmap image raw bytes
        :type bmp_image: numpy.ndarray
        :param fmt: file format, must be one of the supported formats, otherwise the bytes are not saved
        :type fmt: str
        :param fname: file path
        :type fname: pathlib.path
        :param compression: bitmap compression:
            <ul>
                <li> (1) compress width </li>
                <li> (2) compress height </li>
                <li> other values are ignored (no compression) </li>
            </ul>
        :type compression: int
        :return: True if bytes were saved, False otherwise
        :rtype: bool
        """
        bmp_bytes: numpy.ndarray = numpy.array(bmp_image)

        # check for compression
        if compression in (1, 2):

            # convert array values from {0x00, 0xFF} to {0x00, 0x01}
            bmp_bytes = bmp_bytes >= 1

            # create array for compressed bytes
            if compression == 1:

                # compress columns
                compressed_bytes: numpy.ndarray = numpy.zeros(
                    (bmp_bytes.shape[0], ceil(bmp_bytes.shape[1] / 8)),
                    'uint8'
                )

                # compress bytes
                for row in range(bmp_bytes.shape[0]):
                    for col in range(bmp_bytes.shape[1]):
                        shift: int = (8 - (col % 8))
                        compressed_bytes[row][col // 8] |= (bmp_bytes[row][col] << shift) & 0xFF

            else:
                # compress rows
                compressed_bytes: numpy.ndarray = numpy.zeros(
                    (ceil(bmp_bytes.shape[0] / 8), bmp_bytes.shape[1]),
                    'uint8'
                )

                # compress rows
                for row in range(bmp_bytes.shape[0]):
                    shift: int = (row % 8)
                    for col in range(bmp_bytes.shape[1]):
                        compressed_bytes[row // 8][col] |= (bmp_bytes[row][col] << shift) & 0xFF

            bmp_bytes = compressed_bytes

        return self.save_byte_array_to_file(
            bmp_bytes,
            fmt,
            fname.with_name(f"{fname.name}_bmp")
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.image.close()


# ---------------------------------------------------------------------------------------------------------------------


def main() -> int:
    arg_parser = ArgumentParser(
        description="Convert a given image (JPG/JPEG/PNG) into a bitmap/greyscale image, "
                    "and/or extract its bitmap equivalent bytes into a file.",
    )

    # supported operations command group
    oper_group = arg_parser.add_argument_group(
        "operations",
        description="Image conversion operations"
    )

    # supported options command group
    opt_group = arg_parser.add_argument_group(
        "output options",
        description="Image conversion options"
    )

    # input image
    arg_parser.add_argument(
        'in_img',
        help="input image path."
    )

    # output file
    arg_parser.add_argument(
        '-o', '--out_file',
        help="output image/file path. "
             "Default: out",
        default="out"
    )

    # convert to greyscale
    oper_group.add_argument(
        '-g', '--greyscale',
        help="convert image to greyscale and save to output file. "
             "Not enabled by default",
        action="store_true",
        default=False
    )

    # greyscale to monochrome conversion threshold
    opt_group.add_argument(
        '-t', '--threshold',
        help="threshold for converting RGB pixels to bitmap. "
             "Default: 128",
        type=int,
        default=128
    )

    # convert to bitmap
    oper_group.add_argument(
        '-b', '--bitmap',
        help="convert image to bitmap and save to output file. "
             "Not enabled by default",
        action="store_true",
        default=False
    )

    # save raw image bytes
    oper_group.add_argument(
        '-s', '--save',
        help="save raw image bytes to a file. "
             "Not enabled by default",
        action="store_true",
        default=False
    )

    # maximum dimension (for a square image)
    opt_group.add_argument(
        '-m', '--max_dim',
        help="maximum dimension of output image (width or height) in pixels. "
             "Note: this option is overridden by -s if both options are given. "
             "Default: 0",
        type=int,
        default=0
    )

    # size (width x height)
    opt_group.add_argument(
        '-z', '--size',
        help="output image size. Note: this option overrides -m if both options are given. "
             "Default: 0 0 (original image size)",
        nargs=2,
        type=int,
        default=(0, 0)
    )

    # add padding
    opt_group.add_argument(
        '-p', '--padding',
        help="Add padding pixels to resized image to fill remaining image. "
             "Color is added before conversion.",
        type=str,
        default=None
    )

    # bitmap image bytes compression dimension: width [1], height [2]
    opt_group.add_argument(
        '-x', '--compress',
        help="Raw image bytes compression ratio. Only applicable with mono-chrome conversion."
             "Accepted values: (1) compress image width (columns), "
             "(2) compress image height (rows). "
             "Default: 0 (no compression)",
        type=int,
        default=0
    )

    # saved raw bytes format
    opt_group.add_argument(
        '-f', '--format',
        help="Raw image bytes saved file format. "
             "c: for a C header file, with bytes as an array. "
             "b: for raw bytes, "
             "x: for raw hexdump, equivalent to: hexdump -v -e '8/1 \"%%02x\"' -e '\"\\n\"' "
             "Default: h",
        type=str,
        default='h'
    )

    # negative colors
    opt_group.add_argument(
        '-n', '--negative',
        help="Convert image to negative color (invert image colors) before any processing",
        action="store_true",
        default=False
    )

    # parse cmd args
    cmd_args = arg_parser.parse_args(sys.argv[1:])

    # get input image & output file path
    out_file: pathlib.Path = pathlib.Path(cmd_args.out_file).absolute()
    image_file_path: pathlib.Path = pathlib.Path(cmd_args.in_img)

    # check image file
    if not image_file_path.is_file():
        print(f"input file: {cmd_args.in_img} doesn't exist!")
        return -1

    # open image file
    with ImageConverter(image_file_path) as img_converter:

        # get image file from converter
        image_file: Optional[Image.Image] = img_converter.image

        # check file is not None
        if image_file is None:
            print(f"Failed to open image file: {image_file_path}")
            return -1

        # check image extension
        image_extension: Optional[str] = image_file.format
        if image_extension is None:
            print(f"Image extension is unknown: {str(image_file_path)}")
            return -1

        if image_extension.upper() not in ("JPEG", "PNG", "BMP"):
            print(f"Unsupported format: {image_file_path.suffix}. Allowed image formats: JPG/JPEG, PNG")
            return -1

        # check that image has RGB channels
        if image_extension not in ("JPEG", "PNG"):
            print(f"Image {str(image_file_path)} doesn't have RGB channel. Convert it to another format.")
            return -1

        if image_file.mode in ("P", "LAB", "F"):
            print(f"Image {str(image_file_path)} mode: {image_file.mode} is not supported!")
            return -1

        # negative colors
        if cmd_args.negative:
            image_file = ImageOps.invert(image_file)

        # resize image
        if any(cmd_args.size):
            image_file = img_converter.resize_image(image_file, cmd_args.size[0], cmd_args.size[1], cmd_args.padding)

        # set max dimension
        elif cmd_args.max_dim:
            scale: float = cmd_args.max_dim / max(image_file.size)
            width: int = round(scale * image_file.size[0])
            height: int = round(scale * image_file.size[1])
            image_file = img_converter.resize_image(image_file, width, height, cmd_args.padding)

        # check if greyscale image conversion is required
        if cmd_args.greyscale:

            # convert image to grey scale
            gs_image: Image.Image = img_converter.rgb_to_grey(image_file)

            # save greyscale image
            img_converter.save_greyscale_image(gs_image, out_file)

            # save greyscale raw image bytes
            if cmd_args.save:
                img_converter.save_greyscale_bytes(gs_image, cmd_args.format, out_file)

        # check if conversion to bitmap is required
        if cmd_args.bitmap:

            # convert image to bitmap
            bmp_image: Image.Image = img_converter.rgb_to_bitmap(image_file, cmd_args.threshold)

            # save bitmap to file
            img_converter.save_bitmap_image(bmp_image, out_file)

            # save raw bitmap image to a file
            if cmd_args.save:
                ret: bool = img_converter.save_bitmap_bytes(
                    bmp_image,
                    cmd_args.format,
                    out_file,
                    cmd_args.compress
                )

                if not ret:
                    return -1

    return 0


if __name__ == '__main__':
    sys.exit(main())
