"""
Generates an image collage from a directory of PNG images.

This script reads all PNG images from a specified input directory, optionally filters
them by a filename substring, and arranges them into a grid to form one or more
collage images.

Key features:
- Images are padded to the maximum width and height found among the input images
  to ensure uniform cell sizes in the collage before any further resizing.
- Optionally, each image (after padding) can be resized to a specific width before
  being added to the collage (height is auto-scaled to maintain aspect ratio).
- The number of columns in the collage grid is configurable.
- If the number of images exceeds `max_columns * max_rows_per_collage`, multiple
  collage images will be generated, each suffixed with an index (e.g., collage_0.png,
  collage_1.png).

Command-line arguments:
  -i, --input_data_path: Path to the directory containing input PNG images. (required)
  -o, --output_image_path: Full path for the output collage image(s) (must end with .png).
                           If multiple collages are generated, an index will be appended. (required)
  --width: Target width for each individual image in the collage after padding and before
           being placed in the grid. Height will be auto-scaled. (optional, default: None)
  --max_columns: Maximum number of columns in the collage grid. (default: 10)
  --max_rows_per_collage: Maximum number of rows per single output collage image.
                          If None, all images are attempted in one collage, constrained by columns.
                          (optional, default: None)
  --fname_filter: A string to filter input filenames. Only files containing this string
                  will be included in the collage. (optional, default: None)

Example usage (command-line):
  # Create a collage from images in 'input_dir', save as 'collage.png', 5 columns, each image 200px wide
  python make_collage.py -i input_dir -o collages/my_collage.png --max_columns 5 --width 200

  # Create collages with max 3 columns and 4 rows per collage image, from 'source_images'
  python make_collage.py -i source_images -o output/gallery.png --max_columns 3 --max_rows_per_collage 4

  # Create a collage including only images with 'defect' in their filename
  python make_collage.py -i images_folder -o results/defect_collage.png --fname_filter defect
"""
#%%
import glob
import cv2
import os
import numpy as np
from image_utils.img_resize import resize
from gadget_utils.pipeline_utils import fit_array_to_size
import logging

logging.basicConfig(level=logging.INFO) # Configure logging for the module
logger = logging.getLogger(__name__) # Use module-level logger

#%%
def gen_collage(input_path: str,
                output_path_template: str, # Renamed for clarity as it's a template if multiple collages
                max_columns: int,
                target_img_width: int = None, # Renamed for clarity
                max_rows_per_collage: int = None,
                file_filter: str = None) -> None:
    """
    Generates one or more collage images from a directory of PNG files.

    Images are read, padded to a uniform size (max width/height of all images),
    optionally resized to `target_img_width`, and then arranged into a grid.

    Args:
        input_path (str): Path to the directory containing input PNG images.
        output_path_template (str): The base path for the output collage image(s).
                                    Must end with '.png'. If multiple collages are
                                    generated, an index like '_0', '_1' will be
                                    inserted before the .png extension.
        max_columns (int): Maximum number of columns in the collage grid.
        target_img_width (int, optional): Target width for each individual image in the
                                          collage after initial padding. Height is auto-scaled.
                                          If None, images are not resized after padding.
                                          Defaults to None.
        max_rows_per_collage (int, optional): Maximum number of rows per single output
                                              collage image. If None, effectively no row limit
                                              per collage (all images in one, respecting columns).
                                              Defaults to None.
        file_filter (str, optional): A substring to filter filenames. Only files
                                     containing this string will be included.
                                     Defaults to None.

    Raises:
        FileNotFoundError: If `input_path` does not exist or is not a directory.
        ValueError: If `max_columns` is not positive.
    """
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if max_columns <= 0:
        raise ValueError("max_columns must be a positive integer.")

    files = glob.glob(os.path.join(input_path, '*.png'))
    if file_filter is not None:
        files = [s for s in files if file_filter in s]

    if not files:
        logger.info(f"No PNG files found in {input_path} (with filter: {file_filter if file_filter else 'None'}).")
        return

    files = sorted(files)
    logger.debug(f'List of files to process: {files}')

    img_h = []
    img_w=[]
    imgs=[]
    for file in files:
        img=cv2.imread(file)
        img_h.append(img.shape[0])
        img_w.append(img.shape[1])
        imgs.append(img)
    img_h=np.array(img_h)
    img_w=np.array(img_w)
    max_h = 0
    max_w = 0
    if imgs: # Only calculate if there are images
        max_h = img_h.max()
        max_w = img_w.max()
    logger.info(f'[INFO] Max Input Image Height: {max_h}, Max Input Image Width: {max_w}')

    num_total_images = len(imgs)
    if max_rows_per_collage is None or max_rows_per_collage <= 0: # Treat 0 or None as no row limit for a single collage
        max_rows_per_collage = num_total_images # Effectively, all images in one go if columns allow

    images_per_collage = max_columns * max_rows_per_collage
    
    num_collages = (num_total_images + images_per_collage - 1) // images_per_collage # Ceiling division

    output_base, output_ext = os.path.splitext(output_path_template)

    for collage_idx in range(num_collages):
        start_idx = collage_idx * images_per_collage
        end_idx = start_idx + images_per_collage
        current_batch_images = imgs[start_idx:end_idx]

        if not current_batch_images:
            continue

        num_images_in_batch = len(current_batch_images)
        num_rows_for_this_collage = (num_images_in_batch + max_columns - 1) // max_columns

        processed_cells = []
        final_cell_h, final_cell_w = -1, -1

        for img_idx, img_original in enumerate(current_batch_images):
            # Pad to max_h, max_w first
            h_orig, w_orig = img_original.shape[:2]
            pad_h_val = max_h - h_orig
            pad_w_val = max_w - w_orig
            # Ensure non-negative padding; should be if max_h, max_w derived from these images
            pad_h_val = max(0, pad_h_val)
            pad_w_val = max(0, pad_w_val)

            # Symmetrical padding (approx)
            pad_top = pad_h_val // 2
            pad_bottom = pad_h_val - pad_top
            pad_left = pad_w_val // 2
            pad_right = pad_w_val - pad_left

            img_padded = cv2.copyMakeBorder(img_original, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

            # Then resize if target_img_width is specified
            img_cell = img_padded
            if target_img_width is not None:
                img_cell = resize(img_padded, width=target_img_width) # height is auto-scaled by resize

            if final_cell_h == -1: # First image sets the cell dimensions for this collage
                final_cell_h, final_cell_w = img_cell.shape[:2]

            # Ensure all cells are same size (important if resize leads to minor variations, or for last row padding)
            # Using fit_array_to_size to ensure uniform cell size for stacking
            img_cell_uniform, _, _, _, _ = fit_array_to_size(img_cell, W=final_cell_w, H=final_cell_h, value=(0,0,0))
            processed_cells.append(img_cell_uniform)

        # Assemble the collage grid
        collage_rows = []
        for r in range(num_rows_for_this_collage):
            row_start_idx = r * max_columns
            row_images = processed_cells[row_start_idx : row_start_idx + max_columns]

            # Pad the last row if it's not full
            while len(row_images) < max_columns:
                row_images.append(np.zeros((final_cell_h, final_cell_w, 3), dtype=np.uint8)) # Assuming 3 channels

            collage_row = np.hstack(row_images)
            collage_rows.append(collage_row)

        final_collage = np.vstack(collage_rows)

        current_output_path = output_path_template
        if num_collages > 1:
            current_output_path = f"{output_base}_{collage_idx}{output_ext}"

        try:
            cv2.imwrite(current_output_path, final_collage)
            logger.info(f"Saved collage to: {current_output_path}")
        except Exception as e:
            logger.error(f"Error saving collage {current_output_path}: {e}")

# %%
if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Generates an image collage from PNG images in a directory."
    )
    ap.add_argument('-i','--input_data_path',required=True,
                        help='Path to the directory containing input PNG images.')
    ap.add_argument('-o','--output_image_path',required=True,
                        help='Full path for the output collage image(s) (must end with .png). '
                             'If multiple collages, an index is appended (e.g., name_0.png).')
    ap.add_argument('--width',type=int, default=None, # Changed to int, as None is handled
                        help='Target width for each individual image in the collage (height auto-scaled). Default: None (no resize).')
    ap.add_argument('--max_columns',type=int,default=10,
                        help='Maximum number of columns in the collage grid. (default: 10)')
    ap.add_argument('--max_rows_per_collage',type=int, default=None, # Changed to int
                        help='Maximum number of rows per single output collage image. Default: None (all images in one collage if possible).')
    ap.add_argument('--fname_filter',default=None, type=str,
                        help='A string to filter input filenames. Only files containing this string are included. (default: None)')

    args = ap.parse_args()

    input_dir_path = args.input_data_path
    output_img_path_template = args.output_image_path # Renamed for clarity
    max_cols_arg = args.max_columns
    target_width_arg = args.width
    max_rows_arg = args.max_rows_per_collage
    fname_filter_arg = args.fname_filter

    # Validate output path ends with .png
    if not output_img_path_template.lower().endswith('.png'):
        logger.error(f"Output image path must end with .png. Provided: {output_img_path_template}")
        exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_img_path_template)
    if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty (e.g. if path is just 'file.png')
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    elif not output_dir and not os.path.exists(os.getcwd()): # Should not happen usually
         # Fallback if dirname is empty (e.g. output is just 'file.png' in current dir)
        logger.info(f"Output directory is current working directory.")


    try:
        gen_collage(input_dir_path, output_img_path_template, max_cols_arg,
                    target_img_width=target_width_arg,
                    max_rows_per_collage=max_rows_arg,
                    file_filter=fname_filter_arg)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    
    
