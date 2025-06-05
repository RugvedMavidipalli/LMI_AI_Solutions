"""
Flips images in a specified directory either vertically or horizontally.

This script reads all PNG images from an input directory, applies the specified
flip operation (up-down or left-right), and saves the modified images to an
output directory. The output filenames are appended with '_flipud' or '_fliplr'.

Command-line arguments:
  --path_imgs: Path to the directory containing input PNG images. (required)
  --path_out: Path to the directory where flipped images will be saved.
              If it doesn't exist, it will be created. (required)
  --flip: The direction of the flip. Choices: 'ud' (up-down/vertical),
          'lr' (left-right/horizontal). (required)

Example usage (command-line):
  # Flip all PNG images in 'input_folder' horizontally, save to 'output_folder_lr'
  python flip_image.py --path_imgs input_folder --path_out output_folder_lr --flip lr

  # Flip all PNG images in 'input_folder' vertically, save to 'output_folder_ud'
  python flip_image.py --path_imgs input_folder --path_out output_folder_ud --flip ud
"""
import cv2
import os
import argparse
import glob
import numpy as np

BLACK=(0,0,0) # This constant seems unused in the current script logic.

def flip_images(input_path: str, output_path: str, flip_direction: str) -> None:
    """
    Flips all PNG images in a given directory either vertically or horizontally.

    Args:
        input_path (str): The path to the directory containing input PNG images.
        output_path (str): The path to the directory where flipped images will be saved.
                           The directory will be created if it doesn't exist.
        flip_direction (str): The direction of the flip. Must be 'ud' (up-down/vertical)
                              or 'lr' (left-right/horizontal).

    Raises:
        FileNotFoundError: If the input_path directory does not exist.
        ValueError: If an invalid `flip_direction` is provided.
    """
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f'Input image folder does not exist: {input_path}')
    
    if flip_direction not in ['ud', 'lr']:
        raise ValueError("Invalid flip direction. Choose 'ud' for up-down or 'lr' for left-right.")

    img_paths = glob.glob(os.path.join(input_path, '*.png'))
    if not img_paths:
        print(f"No .png files found in {input_path}")
        return
    for path in img_paths:
        im = cv2.imread(path)
        if im is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue

        h,w = im.shape[:2]
        im_name = os.path.basename(path)
        print(f'Processing file: {im_name} with size [{w}x{h}]')

        #flip image
        # For OpenCV, flip codes are: 0 for vertical, 1 for horizontal, -1 for both
        if flip_direction == 'lr': # left-right (horizontal)
            im_out = cv2.flip(im, 1)
            suffix = '_fliplr'
        elif flip_direction == 'ud': # up-down (vertical)
            im_out = cv2.flip(im, 0)
            suffix = '_flipud'
        # np.flip is also fine: axis=1 for lr, axis=0 for ud. cv2.flip is more conventional for CV tasks.
        # im_out = np.flip(im, axis=1 if flip_direction == 'lr' else 0)

        #create output fname
        out_name = os.path.splitext(im_name)[0] + suffix + '.png'
        output_file = os.path.join(output_path, out_name)

        try:
            cv2.imwrite(output_file, im_out)
            print(f'Saved flipped image to: {output_file}')
        except Exception as e:
            print(f"Error saving image {output_file}: {e}")
        print()
    print("Image flipping process complete.")


if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Flips all PNG images in a directory vertically or horizontally.')
    ap.add_argument('--path_imgs', required=True, help='Path to the directory containing input PNG images.')
    ap.add_argument('--path_out', required=True, help='Path to the directory where flipped images will be saved.')
    ap.add_argument('--flip', required=True, choices=['ud','lr'],
                        help="Flip direction: 'ud' for up-down (vertical), 'lr' for left-right (horizontal).")
    args = ap.parse_args()

    output_dir = args.path_out # Use a more descriptive variable name
    if not os.path.isdir(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)
    
    try:
        flip_images(args.path_imgs, output_dir, args.flip)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")