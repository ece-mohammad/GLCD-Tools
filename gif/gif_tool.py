#!/usr/bin/env python3


# ---------------------------------------------------------------------------------------------------------------------

__version__ = "1.3"

# ---------------------------------------------------------------------------------------------------------------------


import logging
import pathlib
import sys
from argparse import ArgumentParser
from binascii import hexlify
from io import StringIO
from math import log10
from typing import List

import numpy
from PIL import Image, ImageOps
from more_itertools import chunked

from GLcdTools.converter.image_converter import ImageConverter

# ------------------------------------------------------------------------------

Logger: logging.Logger = logging.getLogger("LcdTools")


# ------------------------------------------------------------------------------


class GifTool(ImageConverter):
    """
    Extract animated image frames to PNG images and save them
    """

    @staticmethod
    def extract_image_frames(gif_image: Image.Image) -> List[Image.Image]:
        """
        Extract frames from GIF image

        :param gif_image: GIF image object
        :type gif_image: Image.Image
        :return: list of PNG images
        :rtype: List[Image.Image]
        """
        # extract each frame
        image_frames: List[Image.Image] = list()
        while True:
            new_frame: Image.Image = Image.new(mode="RGBA", size=gif_image.size)
            new_frame.paste(gif_image)
            new_frame.info = gif_image.info
            image_frames.append(new_frame)
            try:
                gif_image.seek(gif_image.tell() + 1)
            except EOFError:
                break

        Logger.debug(
            "Extracted %d frames from image %s",
            len(image_frames),
            gif_image.filename if hasattr(gif_image, "filename") else ""
        )

        return image_frames

    @staticmethod
    def get_frames_average_duration_ms(image_frames: List[Image.Image]) -> float:
        """
        Get average frame duration for the frames in the image

        :param image_frames: Animated image
        :type image_frames: Image.Image
        :return: Average frame duration in milliseconds
        :rtype: float
        """

        duration: int = 0
        for (frame_num, frame) in enumerate(image_frames, 1):
            duration += frame.info.get("duration", 0)

        frame_duration: float = duration / len(image_frames)

        Logger.debug("Average frame duration: %f milliseconds", frame_duration)

        return frame_duration

    @staticmethod
    def save_image_frames(image_frames: List[Image.Image], out_dir: pathlib.Path, frame_name: str,
                          ext: str = ".JPG") -> None:
        """
        Save image frames to a given path.

        The frames will be saved as out_dir/file_name_{frame_num}.{ext}

        :param image_frames: a list of image objects as image frames
        :type image_frames: List[Image.Image]
        :param out_dir: path to save images to
        :type out_dir: pathlib.Path
        :param frame_name: name used for saving the frames
        :type frame_name: str
        :param ext: file extension
        :type ext: str
        :return: None
        :rtype: None
        """
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        if not ext.startswith('.'):
            ext = f".{ext}"

        zeros: int = int(log10(len(image_frames))) + 1

        for (frame_num, frame) in enumerate(image_frames, 1):
            frame.save(
                (out_dir / f"{frame_name}_{frame_num:0{zeros}}").with_suffix(ext),
            )

        Logger.debug("Saving %d frames to directory %s", len(image_frames), out_dir)

    def save_frames_bytes_to_file(self, frames: List[numpy.ndarray], out_dir: pathlib.Path, frame_name: str,
                                  fmt: str) -> bool:
        """
        Save frames raw bytes to file. After saving, there will be `N + 1` files (N; number of frames).

        One file will contain the concatenation of all frames together. And N files (a file for each frame).

        Supported formats:
            - c : C array
            - b : binary file
            - h : hexdump

        :param frames: A list of frames as numpy ndarrays, to save their raw bytes
        :type frames: numpy.ndarray
        :param fmt: format to save frames bytes as
        :type fmt: str
        :param out_dir: path to output directory
        :type out_dir: pathlib.Path
        :param frame_name: name used for output files
        :type frame_name: str
        :return: True if saved successfully, False otherwise
        :rtype: bool
        """
        concatenation: List[bytes | str] = list()
        address: int = 0
        num_of_frames: int = len(frames)
        zeros: int = int(log10(num_of_frames)) + 1

        # iterate over each frame, save it & concatenate with previous frames
        for (frame_num, frame) in enumerate(frames, 1):
            file_name: pathlib.Path = out_dir / f"{frame_name}_{frame_num:0{zeros}}"
            self.save_byte_array_to_file(numpy.array(frame), file_name, fmt)
            match fmt:
                case 'b':
                    concatenation.append(frame.tobytes(order='C'))

                case 'x':
                    for chunk in chunked(frame.tobytes(order='C'), 16):
                        with StringIO() as temp:
                            temp.write(f"{address * 16:08X} ")
                            temp.write(f"{hexlify(bytearray(chunk), ' ', 1)}")
                            temp.write("\n")
                            concatenation.append(temp.getvalue())
                        address += 1

                case 'c':
                    with StringIO() as temp:
                        temp.write("    {\n")
                        for chunk in chunked(frame.tobytes(order='C'), frame.shape[1]):
                            temp.write(f"        {{{', '.join([f'0x{i:02x}' for i in chunk])}}},\n")
                        temp.write("    },\n")
                        concatenation.append(temp.getvalue())

                case _:
                    Logger.error("Unsupported format: %s", fmt)
                    return False

        # save concatenation
        file_name = out_dir / frame_name
        match fmt:
            case 'b':
                with open(file_name.with_suffix(".bin"), "wb") as out_file:
                    out_file.writelines(concatenation)
                Logger.debug("Saving frames raw bytes to binary file: %s", file_name)

            case 'x':
                with open(file_name.with_suffix(".hex"), "w") as out_file:
                    out_file.writelines(concatenation)
                Logger.debug("Saving frames raw bytes to hexdump file: %s", file_name)

            case 'c':
                rows, columns = frames[0].shape
                header: str = "#include <stdint.h>\n" \
                              "\n" \
                              "\n" \
                              "#define ROWS      {rows}\n" \
                              "#define COLUMNS   {columns}\n" \
                              "#define FRAMES    {num_of_frames}\n" \
                              "\n" \
                              "\n" \
                              "uint32_t {image_name}_rows = ROWS;\n" \
                              "uint32_t {image_name}_columns = COLUMNS;\n" \
                              "uint32_t {image_name}_frames = FRAMES;\n" \
                              "\n" \
                              "\n" \
                              "uint8_t {image_name}_bytes [FRAMES][ROWS][COLUMNS] = {{\n"

                footer: str = "};\n\n"

                with open(file_name.with_suffix(".c"), "w") as out_file:
                    out_file.write(
                        header.format(
                            rows=rows,
                            columns=columns,
                            num_of_frames=num_of_frames,
                            image_name=frame_name
                        )
                    )

                    concatenation[-1] = concatenation[-1].rstrip().rstrip(',') + '\n'
                    out_file.writelines("".join(concatenation).replace("},\n    }", "}\n    }"))
                    out_file.write(footer)

                Logger.debug("Saved frames raw bytes to C file: %s", file_name)

        return True

    def convert_frames_to_greyscale(self, image_frames: List[Image.Image]) -> List[Image.Image]:
        """
        Convert RGB image frames to greyscale

        :param image_frames: list of image frames
        :type image_frames: List[Image.Image]
        :return: None
        :rtype: None
        """
        grey_frames: List[Image.Image] = list()
        for frame in image_frames:
            grey_frames.append(self.rgb_to_grey(frame))

        Logger.debug("Converting image frames to greyscale")

        return grey_frames

    def convert_frames_to_bitmap(self, image_frames: List[Image.Image], threshold: int = 128) -> List[Image.Image]:
        """
        Convert RGB image frames from to bitmap

        :param image_frames: list of greyscale images
        :type image_frames: Image.Image
        :param threshold: greyscale to monochrome threshold. Default: 128
        :type threshold: int
        :return: None
        :rtype: None
        """
        bitmap_frames: List[Image.Image] = list()
        for frame in image_frames:
            if frame.mode not in ('L', "LA"):
                frame = self.rgb_to_grey(frame)
            bitmap_frames.append(self.greyscale_to_bitmap(frame, threshold=threshold))

        Logger.debug("Converting image frames to bitmap")

        return bitmap_frames

    def save_greyscale_frames(self, image_frames: List[Image.Image], out_dir: pathlib.Path, frame_name: str) -> None:
        """
        Save greyscale image frames to output directory

        :param image_frames: list of greyscale image frames
        :type image_frames: List[Image.Image]
        :param out_dir: output directory path
        :type out_dir: pathlib.Path
        :param frame_name: name used to save each frame
        :type frame_name: str
        :return: None
        :rtype: None
        """
        self.save_image_frames(image_frames, out_dir / f"{frame_name}_grey", f"{frame_name}_gs", ext=".png")

    def save_bitmap_frames(self, image_frames: List[Image.Image], out_dir: pathlib.Path, frame_name: str) -> None:
        """
        Save bitmap image frames

        :param image_frames: list of bitmap frames
        :type image_frames: List[Image.Image]
        :param out_dir: output directory name
        :type out_dir: pathlib.Path
        :param frame_name: name used to save each frame
        :type frame_name: str
        :return: None
        :rtype: None
        """
        self.save_image_frames(image_frames, out_dir / f"{frame_name}_bmp", f"{frame_name}_bmp", ext=".bmp")

    def save_greyscale_frames_bytes(self, image_frames: List[Image.Image], out_dir: pathlib.Path,
                                    frame_name: str, fmt: str) -> bool:
        """
        Save greyscale image frames raw  bytes

        After saving, there will be `N + 1` files (N; number of frames).
        One file containing the concatenation of all frames together. And N files (a file for each frame),
        containing ra bytes of each frame.

        :param image_frames: list of image greyscale frames
        :type image_frames: List[Image.Image]
        :param out_dir: output directory path
        :type out_dir: pathlib.Path
        :param frame_name: name used to save each frame
        :type frame_name: str
        :param fmt: raw bytes file format. Supported formats: {'b': binary file, 'c': C array, 'x': hexdump file}
        :type fmt: str
        :return: True if bytes were saved successfully, False otherwise
        :rtype: bool
        """
        out_dir = out_dir.absolute()
        gs_path: pathlib.Path = out_dir / f"{frame_name}_raw_grey"

        if not gs_path.exists():
            gs_path.mkdir(parents=True, exist_ok=True)

        return self.save_frames_bytes_to_file(
            numpy.stack([numpy.array(frame.getchannel('L')) for frame in image_frames]),
            gs_path,
            frame_name,
            fmt
        )

    def save_bitmap_frames_bytes(self, image_frames: List[Image.Image], out_dir: pathlib.Path, frame_name: str,
                                 fmt: str, compression: int = 0) -> bool:
        """
        Save image bitmap frames raw bytes

        :param image_frames: list of image bitmap frames
        :type image_frames: List[Image.Image]
        :param out_dir: output directory path
        :type out_dir: Path
        :param frame_name: name used to save each frame's bytes
        :type frame_name: str
        :param fmt: raw bytes file format. Supported formats: {'b': binary file, 'c': C array, 'x': hexdump file}
        :type fmt: str
        :param compression: frame bytes compression dimension. 1: compress frame columns (width), 2: compress frame rows (height)
        :type compression: str
        :return: None
        :rtype: None
        """
        out_dir = out_dir.absolute()
        bmp_path: pathlib.Path = out_dir / f"{frame_name}_raw_bmp"

        frames_bytes: List[numpy.ndarray] = [numpy.array(frame) for frame in image_frames]

        for (frame_num, frame) in enumerate(frames_bytes):
            frames_bytes[frame_num] = self.compress_byte_array(frame, compression)

        if not bmp_path.exists():
            bmp_path.mkdir(parents=True, exist_ok=True)

        return self.save_frames_bytes_to_file(
            frames_bytes,
            bmp_path,
            frame_name,
            fmt
        )


def main() -> int:
    arg_parser: ArgumentParser = ArgumentParser(description="Extract frames from animated image.")

    arg_parser.add_argument(
        "in_image",
        help="Extract still frames from a given animated image (GIF), and save them.",
    )

    # operations done on GIF image and frames
    oper_group = arg_parser.add_argument_group("Operations")

    # options used with operations
    opt_group = arg_parser.add_argument_group("Options")

    # output file path
    oper_group.add_argument(
        '-o', '--out_file',
        help="Output file path, default: out",
        default=""
    )

    # greyscale conversion
    oper_group.add_argument(
        '-g', '--greyscale',
        help="Convert extracted frames to bitmap before saving them. Default: False",
        action="store_true",
        default=False
    )

    # bitmap conversion
    oper_group.add_argument(
        '-b', '--bitmap',
        help="Convert extracted frames to bitmap before saving them. Default: False",
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

    # parse cmd arguments
    cmd_args = arg_parser.parse_args(sys.argv[1:])

    # configure logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if cmd_args.verbose else logging.INFO
    )

    # get input image & output file path
    image_path: pathlib.Path = pathlib.Path(cmd_args.in_image).absolute()

    if cmd_args.out_file:
        output_file_path: pathlib.Path = pathlib.Path(cmd_args.out_file)
        if output_file_path.name and output_file_path.parent:
            output_file_name: str = output_file_path.name
            output_file_path = output_file_path.parent
        else:
            output_file_name: str = image_path.stem
            output_file_path = output_file_path.parent
    else:
        output_file_name: str = image_path.stem
        output_file_path = pathlib.Path("out").absolute()

    Logger.info("Output directory: %s", str(output_file_path))

    # open image
    with GifTool(image_path) as gif_tool:

        # get image file from gif tool
        image_file: Image.Image = gif_tool.image

        # check if image file is None
        if image_file is None:
            Logger.critical(f"Filed to open image: %s", str(image_path))
            return -1

        Logger.info("Opening image: %s", str(image_path))

        # extract frames from given image
        image_frames: List[Image.Image] = gif_tool.extract_image_frames(image_file)

        Logger.info(
            "Extracted %d frames from image %s, frame duration %0.2f milliseconds",
            len(image_frames),
            image_path.name,
            gif_tool.get_frames_average_duration_ms(image_frames)
        )

        # loop over extracted frames and transform them as required (invert colors and resize)
        for (frame_number, frame) in enumerate(image_frames, 0):

            # resize frame
            if any(cmd_args.size):
                width: int = cmd_args.size[0]
                height: int = cmd_args.size[1]

            # set frame's max dimension
            elif cmd_args.max_dim:
                scale: float = cmd_args.max_dim / max(frame.size)
                width: int = round(scale * frame.size[0])
                height: int = round(scale * frame.size[1])

            else:
                width, height = frame.size

            frame = gif_tool.resize_image(frame, width, height, cmd_args.padding)

            # invert frame colors
            if cmd_args.negative:
                frame: Image.Image = ImageOps.invert(frame.convert(mode="RGB"))
                if 'A' in image_frames[frame_number].mode:
                    alpha: Image.Image = image_frames[frame_number].getchannel('A')
                    frame = frame.convert(mode="RGBA")
                    frame.putalpha(alpha)

            # put updated frame back in the list
            frame.info = image_frames[frame_number].info
            image_frames[frame_number] = frame

        Logger.info("Frames size: %d %d pixels", image_frames[0].size[0], image_frames[0].size[1])

        # save frames
        gif_tool.save_image_frames(image_frames, output_file_path / f"{output_file_name}_frames", output_file_name,
                                   ext=".png")

        Logger.info(
            "Saving image %s frames to directory %s",
            image_path.name,
            output_file_path / f"{output_file_name}_frames"
        )

        # convert frames to greyscale
        image_frames = gif_tool.convert_frames_to_greyscale(image_frames)
        Logger.info("Converting image frames to greyscale")

        if cmd_args.greyscale:
            gif_tool.save_greyscale_frames(image_frames, output_file_path, output_file_name)
            Logger.info("Saving greyscale frames to directory %s", output_file_path)

            if cmd_args.save:
                gif_tool.save_greyscale_frames_bytes(image_frames, output_file_path, output_file_name, cmd_args.format)
                Logger.info("Saving image greyscale raw bytes to directory %s", output_file_path)

        if cmd_args.bitmap:
            image_frames = gif_tool.convert_frames_to_bitmap(image_frames, cmd_args.threshold)
            Logger.info("Converting image frames to bitmap")

            gif_tool.save_bitmap_frames(image_frames, output_file_path, output_file_name)
            Logger.debug("Saving image bitmap frames to directory %s", output_file_path)

            if cmd_args.save:
                ret: bool = gif_tool.save_bitmap_frames_bytes(
                    image_frames,
                    output_file_path,
                    output_file_name,
                    cmd_args.format,
                    cmd_args.compress
                )

                if ret:
                    Logger.info("saved image bitmap frames raw  bytes to directory %s", output_file_path)
                else:
                    Logger.critical("Failed to save image bitmap raw bytes")
                    return -1

    return 0


if __name__ == '__main__':
    sys.exit(main())
