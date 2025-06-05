"""
Splits images into segments and then stacks these segments.

This script processes PNG images from an input directory. Each image is split
into a specified number of segments, either horizontally or vertically.
- If split horizontally, the segments (columns) are then stacked vertically (np.vstack).
- If split vertically, the segments (rows) are then stacked horizontally (np.hstack).

The output images are saved in a specified directory. Filenames can either be
the same as the original (potentially overwriting if not careful with paths) or
appended with a suffix indicating the split and stack operation.

Command-line arguments:
  --path_imgs: Path to the directory containing input PNG images. (required)
  --path_out: Path to the directory where processed images will be saved.
              If it doesn't exist, it will be created. (required)
  --num_split: The number of segments to split the image into. (required)
  --stack: The direction to stack the segments after splitting. (required)
           Choices: 'h' (horizontal stack of vertical splits),
                    'v' (vertical stack of horizontal splits).
  --keep_same_filename: If specified, saves the output image with the same name as the
                        original. Otherwise, appends a suffix. (optional, default: False)

Example usage (command-line):
  # Split images in 'input_dir' into 3 horizontal segments each, then stack these segments vertically
  python split_stack_image.py --path_imgs input_dir --path_out output_dir --num_split 3 --stack v

  # Split images into 2 vertical segments, stack horizontally, keep original filenames
  python split_stack_image.py --path_imgs input_dir --path_out output_dir --num_split 2 --stack h --keep_same_filename
"""
import cv2
import os
import argparse
import glob
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_vstack_image(im: np.ndarray, num_split: int) -> np.ndarray:
    """
    Splits an image horizontally into `num_split` segments, then stacks them vertically.

    If the image width is not perfectly divisible by `num_split`, a warning is logged,
    and the width of segments is determined by integer division, potentially leaving
    some pixels at the edge if `width % num_split != 0`.

    Args:
        im (np.ndarray): The input image as a NumPy array.
        num_split (int): The number of horizontal segments to create.

    Returns:
        np.ndarray: The image created by vertically stacking the horizontal segments.
                    The output width will be `w // num_split`, and height will be `h * num_split`.

    Raises:
        ValueError: If `num_split` is not positive.
    """
    if num_split <= 0:
        raise ValueError("num_split must be a positive integer.")
    h,w = im.shape[:2]
    if w%num_split:
        logger.warning(f'the image width of {w} is not divisible by {num_split}')
    w_seg = w//num_split
    im_segs = []
    for j in range(num_split):
        s,e = j*w_seg,(j+1)*w_seg
        im_segs.append(im[:,s:e])
    im_out = np.vstack(im_segs)
    return im_out


def split_hstack_image(im: np.ndarray, num_split: int) -> np.ndarray:
    """
    Splits an image vertically into `num_split` segments, then stacks them horizontally.

    If the image height is not perfectly divisible by `num_split`, a warning is logged,
    and the height of segments is determined by integer division.

    Args:
        im (np.ndarray): The input image as a NumPy array.
        num_split (int): The number of vertical segments to create.

    Returns:
        np.ndarray: The image created by horizontally stacking the vertical segments.
                    The output height will be `h // num_split`, and width will be `w * num_split`.

    Raises:
        ValueError: If `num_split` is not positive.
    """
    if num_split <= 0:
        raise ValueError("num_split must be a positive integer.")
    h,w = im.shape[:2]
    if h%num_split:
        logger.warning(f'the image height {h} is not divisible by {num_split}')
    h_seg = h//num_split
    im_segs = []
    for j in range(num_split):
        s,e = j*h_seg,(j+1)*h_seg
        im_segs.append(im[s:e,:])
    im_out = np.hstack(im_segs)
    return im_out


def split_stack_images(input_path: str, output_path: str, num_split: int, stack_direction: str, keep_same_filename: bool = False) -> None:
    """
    Processes all PNG images in an input directory by splitting them into segments
    and then stacking these segments.

    Args:
        input_path (str): Path to the directory containing input PNG images.
        output_path (str): Path to the directory where processed images will be saved.
        num_split (int): The number of segments to split each image into.
        stack_direction (str): Direction for stacking: 'v' for vertical stacking of
                               horizontal splits, 'h' for horizontal stacking of
                               vertical splits.
        keep_same_filename (bool, optional): If True, saves the output image with the
                                             same name as the original. Otherwise, appends
                                             a descriptive suffix. Defaults to False.
    Raises:
        FileNotFoundError: If `input_path` does not exist or is not a directory.
        ValueError: If `stack_direction` is invalid or `num_split` is not positive.
    """
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f'Input image folder does not exist: {input_path}')
    if stack_direction not in ['v', 'h']:
        raise ValueError("stack_direction must be 'v' or 'h'.")
    if num_split <= 0: # This check is also in helper functions, but good for early fail
        raise ValueError("num_split must be a positive integer.")

    img_paths = glob.glob(os.path.join(input_path, '*.png'))
    if not img_paths:
        logger.info(f"No .png files found in {input_path}. Nothing to process.")
        return

    for path in img_paths:
        im = cv2.imread(path)
        if im is None:
            logger.warning(f"Could not read image: {path}. Skipping.")
            continue

        h, w = im.shape[:2]
        im_name = os.path.basename(path)
        logger.info(f'Processing file: {im_name} (Original size: {w}x{h})')

        # split image
        if stack_direction == 'v': # Split horizontally, stack vertically
            im_out = split_vstack_image(im, num_split)
            suffix = f'_split{num_split}_vstack'
        elif stack_direction == 'h': # Split vertically, stack horizontally
            im_out = split_hstack_image(im, num_split)
            suffix = f'_split{num_split}_hstack'

        nh, nw = im_out.shape[:2]
        logger.info(f'Output image size: {nw}x{nh}')

        #create output fname
        base, ext = os.path.splitext(im_name)
        out_name = im_name if keep_same_filename else base + suffix + ext # Use original extension

        output_file_path = os.path.join(output_path, out_name)
        try:
            cv2.imwrite(output_file_path, im_out)
            logger.info(f'Saved processed image to: {output_file_path}\n')
        except Exception as e:
            logger.error(f"Error saving image {output_file_path}: {e}\n")
    logger.info("Image splitting and stacking process complete.")


if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Splits images into segments and stacks them vertically or horizontally.')
    ap.add_argument('--path_imgs', required=True, help='Path to the directory containing input PNG images.')
    ap.add_argument('--path_out', required=True, help='Path to the directory where processed images will be saved.')
    ap.add_argument('--num_split', required=True, type=int, help='The number of segments to split each image into.')
    ap.add_argument('--stack',choices=['h','v'], required=True, # nargs=1 implies a list, so access with args.stack[0]
                        help='Stacking direction after splitting: "h" for horizontal stack (from vertical splits), '
                             '"v" for vertical stack (from horizontal splits).')
    ap.add_argument('--keep_same_filename', action='store_true',
                        help='If set, saves the output image with the same name as the original. Default: appends suffix.')
    args = ap.parse_args()

    output_dir = args.path_out # Use a more descriptive variable name
    if not os.path.isdir(output_dir):
        logger.info(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)
    
    # Correctly access stack direction if nargs=1 (it becomes a list)
    # However, for choices with single selection, nargs is usually not needed or should be `?`
    # For simplicity, assuming stack will be a string as per typical choices.
    # If `nargs=1` is strictly kept, then `args.stack[0]` would be needed.
    # The original code used `args['stack']` which works if `vars(ap.parse_args())` is used.
    # With `ap.parse_args()`, it's `args.stack`. If nargs=1, it's `args.stack[0]`.
    # Let's assume it's intended as a single string choice.
    # The original code `args=vars(ap.parse_args())` and then `args['stack']` would get the list.
    # If `nargs=1` is removed, `args.stack` is a string.
    # Let's assume nargs=1 was intended and use args.stack[0] for robustness.
    stack_choice = args.stack[0] if isinstance(args.stack, list) else args.stack


    try:
        split_stack_images(args.path_imgs, output_dir, args.num_split, stack_choice, args.keep_same_filename)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
