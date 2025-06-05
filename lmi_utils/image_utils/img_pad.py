"""
Pads or crops images to a specified output width and height.

This script processes all images (currently assumes .png, though not explicitly restricted
by `get_relative_paths` which doesn't filter by extension) in an input directory
(optionally recursively) and resizes them to the target dimensions.
If an image is smaller than the target dimensions, it's padded. If larger, it's cropped.
The padding/cropping is done symmetrically from the center using the
`fit_array_to_size` utility.

Output images are saved in a corresponding directory structure under the specified
output path, with filenames appended with '_pad_WxH.png'.

Command-line arguments:
  --path_imgs, -i: Path to the directory containing input images. (required)
  --path_out, -o: Path to the directory where processed images will be saved.
                  If it doesn't exist, it will be created. (required)
  --wh: Target output image size as "width,height" (e.g., "640,480"). (required)
  --recursive: If specified, processes images in subdirectories of `path_imgs` as well.
               (optional, default: False)

Example usage (command-line):
  # Pad/crop all images in 'input_dir' to 800x600, save to 'output_dir_padded'
  python img_pad.py -i input_dir -o output_dir_padded --wh 800,600

  # Process images recursively in 'input_dataset' to 1024x768
  python img_pad.py -i input_dataset -o output_dataset_padded --wh 1024,768 --recursive
"""
import cv2
import os
import argparse
import logging

from gadget_utils.pipeline_utils import fit_array_to_size
from image_utils.path_utils import get_relative_paths


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def fit_image_to_size(input_path: str, output_path: str, out_wh: list[int], recursive: bool) -> None:
    """
    Processes images from an input directory by padding or cropping them to a target size.

    Uses `get_relative_paths` to find images and `fit_array_to_size` to perform
    the actual padding/cropping operation. Output images are saved with a suffix
    indicating the new dimensions.

    Args:
        input_path (str): The path to the directory containing input images.
        output_path (str): The path to the root directory where processed images
                           will be saved. A corresponding subdirectory structure is
                           created if `recursive` is True.
        out_wh (list[int]): A list containing two integers: [target_width, target_height].
        recursive (bool): If True, processes images in subdirectories of `input_path`.

    Raises:
        FileNotFoundError: If the `input_path` directory does not exist.
    """
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f'Input image folder does not exist: {input_path}')

    W, H = out_wh
    img_paths = get_relative_paths(input_path, recursive) # get_relative_paths handles various image types by default
    
    if not img_paths:
        logger.info(f"No images found in {input_path}{' recursively' if recursive else ''}.")
        return
    for relative_img_path in img_paths:
        full_img_path = os.path.join(input_path, relative_img_path)
        im = cv2.imread(full_img_path)

        if im is None:
            logger.warning(f"Could not read image: {full_img_path}. Skipping.")
            continue

        h, w = im.shape[:2]
        im_name = os.path.basename(relative_img_path)
        logger.info(f"Processing: {relative_img_path} (Original size: {w}x{h})")

        #pad image and save it
        im_out, pad_L, pad_R, pad_T, pad_B = fit_array_to_size(im, W, H) # Value defaults to 0 (black)

        if pad_L < 0 or pad_R < 0 or pad_T < 0 or pad_B < 0:
            logger.warning(f'Image {im_name} (original size {w}x{h}) was cropped to fit target size {W}x{H}.')

        #create output fname
        base, ext = os.path.splitext(im_name)
        # Using original extension might be better if not all inputs are PNGs
        out_name = base + f'_pad_{W}x{H}{ext}'

        # Ensure subdirectory structure is replicated in output_path
        output_subdir = os.path.join(output_path, os.path.dirname(relative_img_path))
        if not os.path.isdir(output_subdir):
            os.makedirs(output_subdir)

        output_file_path = os.path.join(output_subdir, out_name)

        try:
            cv2.imwrite(output_file_path, im_out)
            logger.info(f'Successfully saved: {output_file_path}')
        except Exception as e:
            logger.error(f"Error saving image {output_file_path}: {e}")
    logger.info("Image padding/cropping process complete.")


if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Pad or crop images to a specified output size WxH.')
    ap.add_argument('--path_imgs', '-i', required=True, help='Path to the directory containing input images.')
    ap.add_argument('--path_out', '-o', required=True, help='Path to the root directory where processed images will be saved.')
    ap.add_argument('--wh', required=True,
                        help='Target output image size as "width,height" (e.g., "640,480").')
    ap.add_argument('--recursive', action='store_true',
                        help='If set, processes images in subdirectories of path_imgs as well.')
    args = ap.parse_args()

    path_imgs_arg = args.path_imgs
    output_path_arg = args.path_out

    try:
        out_wh_arg = list(map(int, args.wh.split(',')))
        if len(out_wh_arg) != 2:
            raise ValueError("Output size --wh must have two integers (width,height) separated by a comma.")
        if out_wh_arg[0] <= 0 or out_wh_arg[1] <= 0:
            raise ValueError("Output width and height must be positive.")
    except ValueError as e:
        logger.error(f"Invalid format for --wh: {e}")
        exit(1)

    recursive_arg = args.recursive

    logger.info(f'Target output image size: {out_wh_arg[0]}x{out_wh_arg[1]}')

    # Output path creation is handled within fit_image_to_size for subdirectories,
    # but the root output_path should exist or be creatable.
    if not os.path.isdir(output_path_arg):
        try:
            os.makedirs(output_path_arg)
            logger.info(f"Created output directory: {output_path_arg}")
        except OSError as e:
            logger.error(f"Could not create output directory {output_path_arg}: {e}")
            exit(1)
    
    try:
        fit_image_to_size(path_imgs_arg, output_path_arg, out_wh_arg, recursive_arg)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
