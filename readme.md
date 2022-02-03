# GLCD Tools

Small command line tools to work with images for LCD displays. Currently there are 2 tools **image_converter** which
converts images to greyscale and/or monochrome, and **gif_tool** which is the same as *image_converter* but for GIFs; It
extracts frames from animated images and convert them to greyscale/monochrome. In addition to color conversion, both
tools can save the raw bytes for the images (and GIF frames) in different formats: binary, hexdump like file, and C
array.

For more detailed information and usage specifics, check the `README.md` of [image_converter](converter\README.md)
or [gif_tool](gif\README.md)

## Usage

### ImageConverter

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
                        output image/file path. Default: out

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

### GifTool

```text
usage: gif_tool.py [-h] [-o OUT_FILE] [-g] [-b] [-s] [-t THRESHOLD]
                   [-m MAX_DIM] [-z SIZE SIZE] [-p PADDING] [-x COMPRESS]
                   [-f FORMAT] [-n] [-v]
                   in_image

Extract frames from animated image.

positional arguments:
  in_image              Extract still frames from a given animated image
                        (GIF), and save them.

options:
  -h, --help            show this help message and exit

Operations:
  -o OUT_FILE, --out_file OUT_FILE
                        Output file path, default: out
  -g, --greyscale       Convert extracted frames to bitmap before saving them.
                        Default: False
  -b, --bitmap          Convert extracted frames to bitmap before saving them.
                        Default: False
  -s, --save            save raw image bytes to a file. Not enabled by default

Options:
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
  -v, --verbose         Display more debugging information, not enabled by
                        default
```

#### Convert image to greyscale

Convert image `test_image.png` to greyscale and save the greyscale image to `out_gs.png` in the current directory:

`python image_converter.py test_image.png -g`

#### Convert image to monochrome

Convert image `test_image.png` to bitmap and save the bitmap image to `output_dir` as `output_image_bmp.bmp`:

`python image_converter.py test_image.png -b -o output_dir/output_image`

#### Save image greyscale raw bytes to `.bin` file

Convert image `test_image.png` to greyscale and save the greyscale image to same directory as `output_filename_gs.png`.
And save the greyscale image bytes as `output_filename_gs.bin`:

`python image_converter.py test_image.png -g -s -f b -o output_filename`

#### Save image bitmap raw bytes to `C` file

Convert image `test_image.png` to bitmap and save the bitmap image to current directory as `out_bmp.bmp`. And save the
bitmap image bytes as `out_bmp.c`:

`python image_converter.py test_image.png -b -s -f c`

## TODO

- [x] ~~GIFs and animated pictures~~
- [ ] Batch conversions
- [ ] A tool for RGB images (as soon as I get my hands on a color graphical LCD though)
