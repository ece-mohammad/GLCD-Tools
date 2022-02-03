# Image Converter

Convert a JPG/JPEG/PNG (as long as it has separate RGB channels) images to gryscale, bitmap. Can or save the raw image
bytes into file in different formats (C source file, hex dump file, binary file)

## Table of Contents

<!-- MarkdownTOC -->

- [Features](#features)
- [Usage](#usage)
    - [Greyscale conversion](#greyscale-conversion)
    - [Bitmap conversion](#bitmap-conversion)
    - [Image resizing](#image-resizing)
    - [Save raw image bytes](#save-raw-image-bytes)
    - [Image Compression](#image-compression)
- [Requires](#requires)
- [Notes](#notes)

<!-- /MarkdownTOC -->


<a id="features"></a>

## Features

1. Convert PNG/JP(E)G images to greyscale, or bitmap (monochrome), or both at the same time

2. Save intermediate raw image bytes (without image headers), into a file

3. The raw bytes fil can be:
    1. C source file (array + array size)
    2. hexdump file, equivalent to `hexdump -v -e '8/1 "%02x"' -e '"\n"'`
    3. binary file (.bin)

4. Converted image can be resized to a given width x height

5. The image can be resized, such that its max dimension (width or height), does not exceed a given size

6. Raw image bytes can be compressed, either in height (rows), width (columns) or both

7. The image can be padded with given color, if its size is smaller than given value

<a id="usage"></a>

## Usage

```text
usage: image_converter.py [-h] [-o OUT_FILE] [-g] [-b] [-s] [-t THRESHOLD]
                          [-m MAX_DIM] [-z SIZE SIZE] [-p PADDING]
                          [-x COMPRESS] [-f FORMAT] [-n] [-v]
                          in_img

Convert a given image (JPG/JPEG/PNG) into a bitmap/greyscale image, and/or
extract its bitmap equivalent bytes into a file.

positional arguments:
  in_img                input image path.

options:
  -h, --help            show this help message and exit
  -o OUT_FILE, --out_file OUT_FILE
                        path to save output to, in the format: [output directory]/[image name]. Default: out/[image_name]

operations:
  Image conversion operations

  -g, --greyscale       convert image to greyscale and save to output file.
                        Not enabled by default
  -b, --bitmap          convert image to bitmap and save to output file. Not
                        enabled by default
  -s, --save            save raw image bytes to a file. Not enabled by default

output options:
  Image conversion options

  -t THRESHOLD, --threshold THRESHOLD
                        threshold for converting RGB pixels to bitmap.
                        Default: 128
  -m MAX_DIM, --max_dim MAX_DIM
                        maximum dimension of output image (width or height) in
                        pixels. Note: this option is overridden by -s if both
                        options are given. Default: 0
  -z SIZE SIZE, --size SIZE SIZE
                        output image size. Note: this option overrides -m if
                        both options are given. Default: 0 0 (original image
                        size)
  -p PADDING, --padding PADDING
                        Add padding pixels to resized image to fill remaining
                        image. Color is added before conversion.
  -x COMPRESS, --compress COMPRESS
                        Raw image bytes compression ratio. Only applicable
                        with mono-chrome conversion.Accepted values: (1)
                        compress image width (columns), (2) compress image
                        height (rows). Default: 0 (no compression)
  -f FORMAT, --format FORMAT
                        Raw image bytes saved file format. c: for a C header
                        file, with bytes as an array. b: for raw bytes, x: for
                        raw hexdump, equivalent to: hexdump -v -e '8/1 "%02x"'
                        -e '"\n"' Default: h
  -n, --negative        Convert image to negative color (invert image colors)
                        before any processing
  -v, --verbose         Display more debugging information, not enabled by default
```

<a id="greyscale-conversion"></a>

### Greyscale conversion

Convert a given image to greyscale `JPG`, with the name `{output_file_name}_gs.jpg`, `or out_gs.jpg` if no output file
name is give. The image has 1 channel corresponding to luminance, and all its pixels have values in range `[0: 255]`.

- Convert image to greyscale `JPG`, and save it as `out_gs.gpj`:
  `python image_converter.py -g test_image.jpg`

<a id="bitmap-conversion"></a>

### Bitmap conversion

Convert a given image to greyscale, then to a bitmap with the name `{output_file_name}_bmp.bmp`, `or out_gs.bmp` if no
output file name is give. The image has 1 channel corresponding to luminance, and all its pixels have values
of `{0: white pixel, 25: black pixel}`.

##### Examples

- Convert image to bitmap and save it as `out_bmp.bmp`:
  `python image_converter.py -b test_image.jpg`

- Convert image to bitmap with greyscale to bitmap conversion threshold [^1] of 40 , save the bitmap as `out_bmp.bmp`:
  `python image_converter.py -b -t 40 test_image.jpg`

- Convert image to bitmap and save the greyscale and bitmap as `out_image_gs.jpg` and `out_image_bmp.bmp`:python
  `python image_converter.py -b -g -o out_image test_image.jpg`

<a id="image-resizing"></a>

### Image resizing

Image resizing occurs before color conversion, and it can be done through 2 methods:

1. **Fixed width and height**: The image is stretched (or squished) in both directions to fit the given dimensions. As a
   result, the image's width/height ratio may change. To preserve the width/height ratio, color[^2] can be added as a
   padding around the smaller dimension to fill excess area.

2. **Fixed maximum dimension**: The image is scaled in both dimensions, so that its maximum dimension (height or width)
   is equal to the given dimension length.

##### Examples

- Resize image, the image is stretched/squished to fit the given width and height:
  `python image_converter.py -g -b -z 320 240 test_image.jpg`

- Resize image to given width and height, preserving width/height ration, with padding:
  `python image_converter.py -g -b -z 320 240 -p black test_image.jpg`

- Resize image so that its maximum dimension (width or height) fits the given max dimension:
  `python image_converter.py -g -b -m 320 test_image.jpg`

<a id="save-raw-image-bytes"></a>

### Save raw image bytes

Save raw image bytes after converting it to greyscale, or bitmap. The raw bytes does not include image headers and
footers. Can be used with greyscale and bitmap. Saved bytes can be saved in 3 formats:

1. Binary format: Raw image bytes are written to a file, the file can then be loaded with any program that handles
   binary data

2. Hexdump format: Image bytes are written into a file, much like a `hexdump`: 32-bit addresses, 16-bytes separated by
   spaces for each line:

```text
00000040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000050 00 00 00 00 00 00 00 00 00 00 00 ff ff ff ff ff
00000060 ff fe ff fe fe fe fe fe fe fe fe fe fe fe fe fe
00000070 fe fe fe fe fe ff ff ff ff fe fe fe fe fe fe fe
.
.
```

3. C array format: Image bytes are written to a file with the following format:

```C
#include <stdint.h>


#define ROWS      240
#define COLUMNS   320


uint32_t out_bmp_rows = ROWS;
uint32_t out_bmp_columns = COLUMNS;


uint8_t out_bmp_bytes [ROWS][COLUMNS] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, ...
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, ...
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, ...
      .
      .
      .
```

##### Examples

- Convert image to greyscale and bitmap, save raw bytes to binary file:
  `python image_converter.py -g -b -m 320 -s -f b test_image.jpg`

- Convert image to greyscale and bitmap, save raw bytes to hexdumnp file:
  `python image_converter.py -g -b -m 320 -s -f x test_image.jpg`

- Convert image to greyscale and bitmap, save raw bytes to binary file:
  `python image_converter.py -g -b -m 320 -s -f c test_image.jpg`

<a id="image-compression"></a>

### Image Compression

Compress image pixels in along one dimension (width or height), before saving its raw bitmap bytes. Currently,
compression is only used with bitmaps. Since bitmap image pixels values are either `{0, 255}`, each 8 consecutive bytes
can be combined into 1 bit, either long image width or height. Compression in width, combines raw image bytes each 8
columns into 1 column. And compression in height combines each 8 rows into 1 row. The result is a smaller binary file.
The result is the same image but squished image (if viewed in an image viewer), in the compression direction. The binary
file can be used with mono-chrome LCD display, where each pixel is represented by 1 bit, instead of 1 byte.

##### Examples

- Convert image to greyscale and bitmap, save raw bytes to binary file, compress pixels along image width:
  `python image_converter.py -g -b -m 320 -s -f c -x 1 test_image.jpg`

- Convert image to greyscale and bitmap, save raw bytes to binary file, compress pixels along image height:
  `python image_converter.py -g -b -m 320 -s -f c -x 2 test_image.jpg`

<a id="requires"></a>

## Requires

1. Python3 (version 3.10)
2. Numpy\~=1.22.1
3. Pillow\~=9.0.0

<a id="notes"></a>

# Notes

[^1]: Greyscale to bitmap conversion threshold is used to convert image pixels from greyscale to monochrome (black and
white only), by converting all pixels with value less than the threshold to black and pixels with values larger than or
equal to white.

[^2]: Color can be an HTML color name (string), RGB function `rgb(R, G, B)`, HSV function `hsv(H, S, V)`, HSL
function `hsl(H, S, L)` or hex encoded color `#RGB, #RRGGBB`. If funtions are used, remember to enclose color in double
quotes to escape spaces 
