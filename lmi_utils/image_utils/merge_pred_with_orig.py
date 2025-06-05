import cv2
"""
Merges pairs of images (e.g., an original image and its corresponding prediction)
by horizontally stacking them.

This script takes two input directories: one for original images and one for
prediction images. It assumes a one-to-one correspondence between images in
these directories, based on sorted filenames. The images must have the same
dimensions for stacking. The merged images (original on the left, prediction
on the right) are saved to an output directory.

Command-line arguments:
  --path_orig: Path to the directory containing original images. (required)
  --path_pred: Path to the directory containing prediction images. (required)
  -o, --path_out: Path to the directory where merged images will be saved.
                  If it doesn't exist, it will be created. (required)
  --fmt: The file format (extension) of the images to process. (default: 'png')

Example usage (command-line):
  # Merge images from 'originals' and 'predictions' folders, save to 'merged_output'
  python merge_pred_with_orig.py --path_orig originals --path_pred predictions -o merged_output --fmt jpg
"""
import cv2
import os
import argparse
import glob
import numpy as np # For type hinting

# import gadget_utils.pipeline_utils as pipeline_utils # Unused import

# BLACK=(0,0,0) # Unused constant

def hstack_imgs(orig_folder: str, pred_folder: str, fmt: str, output_path: str) -> None:
    """
    Horizontally stacks pairs of images from two folders and saves the results.

    Images are matched based on sorted filenames from `orig_folder` and `pred_folder`.
    It's crucial that filenames correspond correctly after sorting for meaningful pairs.
    Both images in a pair must have the same height and number of channels.

    Args:
        orig_folder (str): Path to the directory with original images.
        pred_folder (str): Path to the directory with prediction images.
        fmt (str): File format (extension) of the images (e.g., 'png', 'jpg').
        output_path (str): Path to the directory where merged images will be saved.

    Raises:
        FileNotFoundError: If `orig_folder` or `pred_folder` does not exist.
        AssertionError: If the number of images in the two folders does not match,
                        or if a pair of images does not have the same shape.
    """
    if not os.path.isdir(orig_folder):
        raise FileNotFoundError(f"Original images folder not found: {orig_folder}")
    if not os.path.isdir(pred_folder):
        raise FileNotFoundError(f"Prediction images folder not found: {pred_folder}")

    orig_files = sorted(glob.glob(os.path.join(orig_folder, f'*.{fmt}')))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, f'*.{fmt}')))
    
    if not orig_files:
        print(f"No images with format '.{fmt}' found in {orig_folder}")
        return
    if not pred_files:
        print(f"No images with format '.{fmt}' found in {pred_folder}")
        return

    assert len(orig_files) == len(pred_files), \
        f"The number of images in the two folders must be the same. " \
        f"Found {len(orig_files)} in '{orig_folder}' and {len(pred_files)} in '{pred_folder}'."

    print(f"Found {len(orig_files)} pairs of images to merge.")

    for i, (orig_file_path, pred_file_path) in enumerate(zip(orig_files, pred_files)):
        orig_im = cv2.imread(orig_file_path)
        pred_im = cv2.imread(pred_file_path)

        if orig_im is None:
            print(f"Warning: Could not read original image {orig_file_path}. Skipping pair {i+1}.")
            continue
        if pred_im is None:
            print(f"Warning: Could not read prediction image {pred_file_path}. Skipping pair {i+1}.")
            continue

        # For hconcat, heights and number of channels must be the same. Widths can differ.
        # If strict shape matching (including width) is desired, use:
        # assert orig_im.shape == pred_im.shape, \
        #    f'The shape of {orig_file_path} ({orig_im.shape}) and {pred_file_path} ({pred_im.shape}) must be the same.'
        if orig_im.shape[0] != pred_im.shape[0] or orig_im.shape[2] != pred_im.shape[2]:
             print(f"Warning: Height or channel mismatch for pair: '{os.path.basename(orig_file_path)}' ({orig_im.shape}) "
                   f"and '{os.path.basename(pred_file_path)}' ({pred_im.shape}). Skipping this pair.")
             continue

        try:
            hstack_im = cv2.hconcat([orig_im, pred_im])
            output_filename = os.path.join(output_path, os.path.basename(orig_file_path))
            cv2.imwrite(output_filename, hstack_im)
            print(f"Saved merged image: {output_filename}")
        except Exception as e:
            print(f"Error processing pair {os.path.basename(orig_file_path)} and "
                  f"{os.path.basename(pred_file_path)}: {e}")

    print("Image merging process complete.")


if __name__=="__main__":
    ap = argparse.ArgumentParser(
        description="Horizontally stacks pairs of original and prediction images from two folders."
    )
    ap.add_argument('--path_orig', required=True, help='Path to the directory containing original images.')
    ap.add_argument('--path_pred', required=True, help='Path to the directory containing prediction images.')
    ap.add_argument('-o', '--path_out', required=True, help='Path to the directory where merged images will be saved.')
    ap.add_argument('--fmt', default='png', help="The file format (extension) of the images (default: 'png').")
    args = ap.parse_args()

    output_dir = args.path_out # Use a more descriptive variable name
    if not os.path.isdir(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)
    
    try:
        hstack_imgs(args.path_orig, args.path_pred, args.fmt, output_dir)
    except (FileNotFoundError, AssertionError, Exception) as e:
        print(f"Error: {e}")
