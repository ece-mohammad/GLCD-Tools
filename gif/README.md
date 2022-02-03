# GIF tool

Extract frames from images, convert them to greyscale and/or bitmap and save them. Can save the raw bytes of the image (
separately, and aggregated into one file).

## Usage

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
