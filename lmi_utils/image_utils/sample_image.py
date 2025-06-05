"""
Samples a specified number of PNG images from a source directory and copies
them to a destination directory.

This script allows for either random sampling of images or selecting the
first N images (based on sorted filenames) from the source directory.

Command-line arguments:
  --path_imgs, -i: Path to the directory containing input PNG images. (required)
  --path_out, -o: Path to the directory where sampled images will be copied.
                  If it doesn't exist, it will be created. (required)
                  Must be different from `path_imgs`.
  --num_samples, -n: The number of images to sample. (required)
  --random: If specified, samples images randomly. Otherwise, takes the first
            `num_samples` images after sorting filenames. (optional, default: False)

Example usage (command-line):
  # Randomly sample 50 PNG images from 'all_images' to 'sampled_set'
  python sample_image.py -i all_images -o sampled_set -n 50 --random

  # Copy the first 100 PNG images (alphabetically sorted) from 'dataset' to 'first_100'
  python sample_image.py -i dataset -o first_100 -n 100
"""
#built-in packages
import os
import glob
import logging

#3rd party packages
import shutil
import random

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Sets logging level for this module

def sample_images(path_imgs: str, path_out: str, num_samples: int, is_random: bool = False) -> None:
    """
    Samples a specified number of PNG images from a source directory and copies them
    to a destination directory.

    Args:
        path_imgs (str): The path to the directory containing input PNG images.
        path_out (str): The path to the directory where sampled images will be copied.
        num_samples (int): The number of images to sample.
        is_random (bool, optional): If True, samples images randomly. Otherwise,
                                    takes the first `num_samples` images based on
                                    sorted filenames. Defaults to False.

    Raises:
        FileNotFoundError: If `path_imgs` does not exist or is not a directory.
        ValueError: If `num_samples` is not a positive integer.
    """
    if not os.path.isdir(path_imgs):
        raise FileNotFoundError(f"Input image directory not found: {path_imgs}")
    if num_samples <= 0:
        raise ValueError("Number of samples must be a positive integer.")

    file_list = sorted(glob.glob(os.path.join(path_imgs, '*.png'))) # Sort for consistent non-random sampling

    if not file_list:
        logger.info(f"No .png files found in {path_imgs}. Nothing to sample.")
        return

    if is_random:
        random.shuffle(file_list)
    
    actual_num_to_sample = min(num_samples, len(file_list))
    if actual_num_to_sample < num_samples:
        logger.warning(f"Requested {num_samples} samples, but only {len(file_list)} images are available. "
                       f"Sampling {actual_num_to_sample} images instead.")

    sampled_files = file_list[:actual_num_to_sample]

    copied_count = 0
    for file_path in sampled_files:
        im_name = os.path.basename(file_path)
        dest_file_path = os.path.join(path_out, im_name)
        try:
            shutil.copy2(file_path, dest_file_path) # copy2 preserves metadata
            logger.info(f'Copied: {im_name} to {dest_file_path}')
            copied_count +=1
        except Exception as e:
            logger.error(f"Error copying file {file_path} to {dest_file_path}: {e}")

    logger.info(f'Successfully copied {copied_count} out of {actual_num_to_sample} selected images to {path_out}\n')
    return


if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Samples a specified number of PNG images from a source directory to a destination directory."
    )
    ap.add_argument('--path_imgs', '-i', required=True, help='Path to the directory containing input PNG images.')
    ap.add_argument('--path_out', '-o', required=True, help='Path to the directory where sampled images will be copied.')
    ap.add_argument('--num_samples', '-n', required=True, type=int, help='The number of images to sample.')
    ap.add_argument('--random', action='store_true', help='If set, samples images randomly. Default is to take first N sorted images.')
    args = ap.parse_args()

    # Validate paths
    if args.path_imgs == args.path_out:
        logger.error('Input and output paths must be different.')
        exit(1)

    if not os.path.isdir(args.path_imgs): # Redundant if FileNotFoundError is caught, but good for early exit in CLI
        logger.error(f"Input image directory not found: {args.path_imgs}")
        exit(1)

    # Create output path if it doesn't exist
    if not os.path.isdir(args.path_out):
        try:
            os.makedirs(args.path_out)
            logger.info(f"Created output directory: {args.path_out}")
        except OSError as e:
            logger.error(f"Could not create output directory {args.path_out}: {e}")
            exit(1)

    try:
        sample_images(args.path_imgs, args.path_out, args.num_samples, args.random)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
