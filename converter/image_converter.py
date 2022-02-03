#!/usr/bin/env python3

# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------

__version__ = "1.4"

# ---------------------------------------------------------------------------------------------------------------------
import logging
import pathlib
import sys
from argparse import ArgumentParser
from binascii import hexlify
from io import StringIO
from math import ceil
from typing import Optional, Tuple

import numpy
from PIL import Image, ImageOps
from more_itertools import chunked

# ------------------------------------------------------------------------------

Logger: logging.Logger = logging.getLogger("LcdTools")


# ------------------------------------------------------------------------------


class ImageConverter(object):

    def __init__(self, image_path: Optional[pathlib.Path] = None):
        self.image_path: Optional[pathlib.Path] = image_path
        self.image: Optional[Image.Image] = self.open_image(self.image_path)

    @staticmethod
    def check_image_file(image_file: pathlib.Path) -> bool:
        """
        check if image file path exists and is a file

        :param image_file: path to image file
        :type image_file: pathlib.Path
        :return: True if file exists, False otherwise
        :rtype: None
        """
        return image_file.exists() and image_file.is_file()

    @staticmethod
    def resize_image(image: Image.Image, width: int, height: int, padding: Optional[str] = None) -> Image.Image:
        """
        Resize given image to the given width and height.

        The image will be stretched or squished in x, y directions to fill the given width and height.
        However, If a padding color was specified, the image is scaled to fill width and height without distortion,
        preserving the width/height ratio and the remaining pixels are filled with padding color.

        :param image: image object to resize
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

        if image.size == (width, height):
            return image

        if padding is None:
            Logger.debug(
                "Resizing image %s to %d x %d (no padding)",
                image.filename if hasattr(image, "filename") else "",
                width,
                height
            )
            return image.resize((width, height))
        else:
            Logger.debug(
                "Resizing image %s to %d x %d, padding coloe: s",
                image.filename if hasattr(image, "filename") else "",
                width,
                height,
                padding
            )
            return ImageOps.pad(image, (width, height), color=padding, centering=(0.5, 0.5))

    @staticmethod
    def scale_image(image: Image.Image, scale: float) -> Image.Image:
        """
        resize image in both directions (x, y) with given scale. Note that scaling up uses more memory (RAM).

        :param image: image object to scale
        :type image: Image.Image
        :param scale: scale applied to image dimensions, 0 < scale < 1 to scale down, 1 < scale < inf to scale up
        :type scale: float
        :return: scaled image
        :rtype: Image.Image
        """
        if scale <= 0:
            return image
        Logger.debug("Scaling image %s using scale: %f", image.filename if hasattr(image, "filename") else "", scale)
        return ImageOps.scale(image, scale)

    @staticmethod
    def save_byte_array_to_file(byte_array: numpy.ndarray, file_name: pathlib.Path, fmt: str) -> bool:
        """
        Save given byte array into a file.

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param byte_array: raw bytes as a numpy array
        :type byte_array: numpy.ndarray
        :param fmt: file format, must be one of the supported formats, otherwise the bytes are not saved
        :type fmt: str
        :param file_name: file path
        :type file_name: pathlib.Path
        :return: True if bytes were saved, False otherwise
        :rtype: bool
        """
        match fmt:
            # save as binary file
            case 'b':
                file_name = file_name.with_suffix(".bin")
                byte_array.tofile(file_name, sep="", format="%02x")
                Logger.debug(f"Saved raw bytes as binary file: %s", str(file_name))

            # save as hexdump
            case 'x':
                file_name = file_name.with_suffix(".hex")
                with open(file_name, "wb") as bin_file:
                    for address, chunk in enumerate(chunked(byte_array.tobytes(order='C'), 16)):
                        bin_file.write(bytes(f"{address * 16:08X} ", "ascii"))
                        bin_file.write(hexlify(bytearray(chunk), ' ', 1))
                        bin_file.write(b'\n')
                Logger.debug("Saved raw bytes as hexdump file: %s", str(file_name))

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

                file_name = file_name.with_suffix(".c")
                with open(file_name, "w") as c_file, StringIO() as temp_file:
                    for chunk in chunked(byte_array.tobytes(order='C'), columns):
                        temp_file.write(
                            f"    {{{', '.join([f'0x{i:02x}' for i in chunk])}}},\n"
                        )

                    c_file.write(
                        header.format(
                            rows=rows,
                            columns=columns,
                            image_name=file_name.name
                        )
                    )
                    c_file.write(f"{temp_file.getvalue().rstrip().rstrip(',')}\n")
                    c_file.write(footer)
                Logger.debug("Saved raw bytes as C array to file: %s", str(file_name))

            case _:
                Logger.error("Unsupported format %s", fmt)
                return False

        return True

    @staticmethod
    def save_image(image: Image.Image, file_name: pathlib.Path, suffix: str = "", ext: str = ".JPG") -> None:
        """
        Save image as file name, with optional suffix and extension

        :param image: Image object to save
        :type image: Image.Image
        :param file_name: path to save image to
        :type file_name: pathlib.Path
        :param suffix: suffix added to filename before extension
        :type suffix: str
        :param ext: saved image extension. default: ".JPG"
        :type ext: str
        :return: None
        :rtype: None
        """
        file_name = file_name.with_stem(f"{file_name.stem}{suffix}").with_suffix(ext)
        Logger.debug("Saving image to file: %s", str(file_name))
        return image.save(file_name)

    @staticmethod
    def rgb_to_grey(image: Image.Image) -> Image.Image:
        """
        Convert RGB image to greyscale without loosing transparency (alpha channel)

        :param image: Image object
        :type image: Image.Image
        :return: Greyscale image
        :rtype: Image.Image
        """
        if image.mode in ("L", "LA"):
            return image
        grey: Image.Image = image.convert(mode="LA")
        grey.info = image.info
        Logger.debug("Converting image %s to greyscale", image.filename if hasattr(image, "filename") else "")
        return grey

    @staticmethod
    def compress_byte_array(byte_array: numpy.ndarray, compression: int = 1, threshold: int = 1) -> numpy.ndarray:
        """
        Compress a 2D byte array.

        First, the byte array values are converted to 0s and 1s using the threshold.
        Then each 8 bytes are grouped into 1 byte depending on the compression dimension.
        The compressed array will have a smaller number of columns or rows, depending on the compressed dimension.
        Compression dimension `1` compresses array columns. And compression dimension `2` compresses array rows.

        :param byte_array: byte array to compress
        :type byte_array: numpy.ndarray
        :param compression: compression dimension. (1) compress array columns, (2): compress array rows
        :type compression: int
        :param threshold: value threshold
        :type threshold: int
        :return: compressed byte array if compression is in (1, 2). Otherwise, the original array is returned
        :rtype: numpy.ndarray
        """
        # check for compression
        if compression == 0:
            return byte_array

        if compression not in (1, 2):
            Logger.error("Unsupported compression dimension: %d", compression)
            return byte_array

        # convert array values to {0x00, 0x01}
        byte_array = byte_array >= threshold

        # create array for compressed bytes
        if compression == 1:

            # compress columns
            compressed_bytes: numpy.ndarray = numpy.zeros(
                (byte_array.shape[0], ceil(byte_array.shape[1] / 8)),
                'uint8'
            )

            # compress bytes
            for row in range(byte_array.shape[0]):
                for col in range(byte_array.shape[1]):
                    shift: int = (8 - (col % 8))
                    compressed_bytes[row][col // 8] |= (byte_array[row][col] << shift) & 0xFF

            Logger.debug("Compressing byte array columns")

        else:
            # compress rows
            compressed_bytes: numpy.ndarray = numpy.zeros(
                (ceil(byte_array.shape[0] / 8), byte_array.shape[1]),
                'uint8'
            )

            # compress rows
            for row in range(byte_array.shape[0]):
                shift: int = (row % 8)
                for col in range(byte_array.shape[1]):
                    compressed_bytes[row // 8][col] |= (byte_array[row][col] << shift) & 0xFF

            Logger.debug("Compressing byte array rows")

        return compressed_bytes

    def save_greyscale_image(self, gs_image: Image.Image, image_path: pathlib.Path) -> None:
        """
        Save greyscale image as {image_path}_gs.png

        :param gs_image: greyscale image object
        :type gs_image: Image.Image
        :param image_path: path to save image to
        :type image_path: pathlib.Path
        :return: None
        :rtype: None
        """
        self.save_image(gs_image, image_path, suffix="_gs", ext=".png")

    def save_bitmap_image(self, bmp_image: Image.Image, image_path: pathlib.Path) -> None:
        """
        Save bitmap image as {image_path}_bmp.bmp

        :param bmp_image: bitmap image
        :type bmp_image: Image.Image
        :param image_path: bitmap image path
        :type image_path: pathlib.Path
        :return: None
        :rtype: None
        """
        self.save_image(bmp_image, image_path, suffix="_bmp", ext=".bmp")

    def greyscale_to_bitmap(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Convert image to bitmap

        :param image: RGB image object
        :type image: Image.Image
        :param threshold: Value that separates black/white pixels. Pixels with value >= threshold will be white.
        :type threshold: int
        :return: Image.Image
        :rtype: Image.Image
        """
        grey: Image.Image = self.rgb_to_grey(image)
        bitmap_bytes: numpy.ndarray = numpy.array(grey.getchannel('L'))
        bitmap_bytes: numpy.ndarray = numpy.floor(bitmap_bytes / threshold) * 255
        bitmap_bytes = bitmap_bytes.astype('uint8')
        bitmap: Image.Image = Image.fromarray(bitmap_bytes, mode='L')
        bitmap.info = image.info
        Logger.debug("Converted image %s to bitmap", image.filename if hasattr(image, "filename") else "")
        return bitmap

    def open_image(self, image_file: Optional[pathlib.Path]) -> Optional[Image.Image]:
        """
        Open image file, and return the open image as PIL.Image object

        :param image_file: path to image file
        :type image_file: pathlib.Path
        :return: opened image file if it exists, otherwise return None
        :rtype: Image
        """
        image: Optional[Image.Image] = None
        if image_file is not None and self.check_image_file(image_file):
            try:
                image = Image.open(image_file)
                Logger.debug("Opened image: %s", str(image_file))
            except IOError as ioe:
                Logger.error("Failed to open image: %s", str(image_file))
        return image

    def save_greyscale_bytes(self, gs_image: Image.Image, fmt: str, file_name: pathlib.Path) -> bool:
        """
        Save greyscale image raw bytes to a file

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param gs_image: greyscale image raw bytes
        :type gs_image: numpy.ndarray
        :param fmt: file format, must be one of the supported formats, otherwise the bytes are not saved
        :type fmt: str
        :param file_name: file path
        :type file_name: pathlib.Path
        :return: True if bytes were saved, False otherwise
        :rtype: bool
        """
        return self.save_byte_array_to_file(
            numpy.array(gs_image.getchannel('L')).astype('uint8'),
            file_name.with_name(f"{file_name.name}_gs"),
            fmt
        )

    def save_bitmap_bytes(self, bmp_image: Image.Image, fmt: str, file_name: pathlib.Path,
                          compression: int = 0) -> bool:
        """
        Save bitmap image raw bytes to a file, and apply compression

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param bmp_image: bitmap image raw bytes
        :type bmp_image: numpy.ndarray
        :param fmt: file format, must be one of the supported formats, otherwise the bytes are not saved
        :type fmt: str
        :param file_name: file path
        :type file_name: pathlib.path
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
        bmp_bytes: numpy.ndarray = self.compress_byte_array(numpy.array(bmp_image), compression)
        return self.save_byte_array_to_file(
            bmp_bytes,
            file_name.with_name(f"{file_name.name}_bmp"),
            fmt,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image is not None:
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
        help="input image path.",
    )

    # output file
    arg_parser.add_argument(
        '-o', '--out_file',
        help="path to save output to, in the format: [output directory]/[image name]"
             "Default: out/[image_name]",
        default=""
    )

    # convert to greyscale
    oper_group.add_argument(
        '-g', '--greyscale',
        help="convert image to greyscale and save to output file. "
             "Not enabled by default",
        action="store_true",
        default=False
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

    # greyscale to monochrome conversion threshold
    opt_group.add_argument(
        '-t', '--threshold',
        help="threshold for converting RGB pixels to bitmap. "
             "Default: 128",
        type=int,
        default=128
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

    opt_group.add_argument(
        '-v', '--verbose',
        help="Display more debugging information, not enabled by default",
        action="store_true",
        default=False
    )

    # parse cmd args
    cmd_args = arg_parser.parse_args(sys.argv[1:])

    # configure logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if cmd_args.verbose else logging.INFO
    )

    # image path
    image_file_path: pathlib.Path = pathlib.Path(cmd_args.in_img).absolute()

    # get output file path
    if cmd_args.out_file:
        out_file: pathlib.Path = pathlib.Path(cmd_args.out_file).absolute()
    else:
        out_file: pathlib.Path = pathlib.Path("out") / image_file_path.name

    # check if output parent directory exists
    out_dir: pathlib.Path = out_file.parent
    if not out_dir.exists():
        Logger.debug("Creating output directory: %s", str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

    Logger.info("Output directory: %s", str(out_dir))

    # open image file
    with ImageConverter(image_file_path) as img_converter:

        # get image file from converter
        image_file: Optional[Image.Image] = img_converter.image

        # check image is not None
        if image_file is None:
            Logger.critical("Failed to open image file: %s", image_file_path)
            return -1

        # check image mode
        if image_file.mode in ("P", "LAB", "F"):
            Logger.critical(f"Image %s mode: %s is not supported!", str(image_file_path), image_file.mode)
            return -1

        # resize image
        if any(cmd_args.size):
            width: int = cmd_args.size[0]
            height: int = cmd_args.size[1]
            Logger.info("Resizing image %s to %d x %d pixels", image_file_path.name, width, height)

        # set max dimension
        elif cmd_args.max_dim:
            scale: float = cmd_args.max_dim / max(image_file.size)
            width: int = round(scale * image_file.size[0])
            height: int = round(scale * image_file.size[1])
            Logger.info("Scaling image %s to %d x %d pixels", image_file_path.name, width, height)

        else:
            width, height = image_file.size

        # resize image
        image_file = img_converter.resize_image(image_file, width, height, cmd_args.padding)

        # negative colors
        if cmd_args.negative:
            negative: Image.Image = ImageOps.invert(image_file.convert("RGB"))
            if 'A' in image_file.mode:
                alpha: Image.Image = image_file.getchannel('A')
                negative.putalpha(alpha)
            image_file.paste(negative)
            Logger.info("Inverting image %s colors", image_file_path.name)

        Logger.info(
            "Image info:\nname: %s\nsize:%d, %d\nmode: %s",
            image_file_path.name,
            image_file.size[0],
            image_file.size[1],
            image_file.mode
        )

        # convert image to grey scale
        gs_image: Image.Image = img_converter.rgb_to_grey(image_file)
        Logger.info("Converting image %s to greyscale", image_file_path.name)

        # check if greyscale image conversion is required
        if cmd_args.greyscale:

            # save greyscale image
            img_converter.save_greyscale_image(gs_image, out_file)
            Logger.info("Saving image as greyscale: %s", image_file_path.name)

            # save greyscale raw image bytes
            if cmd_args.save:
                img_converter.save_greyscale_bytes(gs_image, cmd_args.format, out_file)
                Logger.info("Saving image %s greyscale raw bytes", image_file_path.name)

        # check if conversion to bitmap is required
        if cmd_args.bitmap:

            # convert image to bitmap
            bmp_image: Image.Image = img_converter.greyscale_to_bitmap(gs_image, cmd_args.threshold)
            Logger.info("Converting image %s to bitmap", image_file_path.name)

            # save bitmap to file
            img_converter.save_bitmap_image(bmp_image, out_file)
            Logger.info("Saving image %s as bitmap", image_file_path.name)

            # save raw bitmap image to a file
            if cmd_args.save:
                ret: bool = img_converter.save_bitmap_bytes(
                    bmp_image,
                    cmd_args.format,
                    out_file,
                    cmd_args.compress
                )

                if not ret:
                    Logger.critical("Failed to save image %s bitmap raw bytes", image_file_path.name)
                    return -1
                else:
                    Logger.info("Saved image %s bitmap raw bytes", image_file_path.name)

    return 0


if __name__ == '__main__':
    sys.exit(main())
