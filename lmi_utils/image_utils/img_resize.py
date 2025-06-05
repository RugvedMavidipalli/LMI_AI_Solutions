"""
Resizes images from an input directory to specified dimensions.

This script processes images (e.g., .png, .jpg) from an input directory,
optionally recursively. It can resize to a target width, height, or both.
If only one dimension (width or height) is provided, the other is scaled to
maintain the original aspect ratio.

The script can optionally leverage CUDA-enabled OpenCV for resizing if a CUDA
device is available and OpenCV was built with CUDA support.

Output images are saved in a corresponding directory structure under the specified
output path, with filenames appended with '_resize_WxH.png' (where W and H are
the output dimensions, or 'w'/'h' if one was auto-calculated).

Command-line arguments:
  -i, --input_path: Path to the directory containing input images. (required)
  -o, --output_path: Path to the directory where resized images will be saved.
                     If it doesn't exist, it will be created. (required)
  --width: Target width for resizing. If not provided, it's calculated from
           height to maintain aspect ratio, or original width is used if
           height is also not provided. (optional)
  --height: Target height for resizing. If not provided, it's calculated from
            width to maintain aspect ratio, or original height is used if
            width is also not provided. (optional)
  --recursive: If specified, processes images in subdirectories as well.
               (optional, default: False)

Example usage (command-line):
  # Resize all images in 'input_dir' to a width of 800px (height auto-scaled)
  python img_resize.py -i input_dir -o output_dir_resized --width 800

  # Resize all images in 'input_dir' to a height of 600px (width auto-scaled)
  python img_resize.py -i input_dir -o output_dir_resized --height 600

  # Resize all images in 'input_dir' to 1024x768
  python img_resize.py -i input_dir -o output_dir_resized --width 1024 --height 768 --recursive
"""
import numpy as np
import cv2
import os
import logging

from image_utils.path_utils import get_relative_paths

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def is_cuda_cv() -> bool:
    """
    Checks if OpenCV is compiled with CUDA support and a CUDA-enabled device is available.

    Returns:
        bool: True if CUDA is available and enabled in OpenCV, False otherwise.
    """
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return True # CUDA device found
        else:
            return False # No CUDA device found
    except AttributeError: # cv2.cuda module not found (OpenCV not compiled with CUDA)
        return False
    except Exception: # Other potential errors
        return False


def resize(image: np.ndarray, width: int = None, height: int = None, device: str = 'cpu', inter: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resizes an image to specified dimensions, preserving aspect ratio if one dimension is omitted.

    Can use CPU or GPU (if CUDA is available in OpenCV).

    Args:
        image (np.ndarray): The input image as a NumPy array.
        width (int, optional): The target width. If None, it's calculated from `height`
                               to maintain aspect ratio, or original width is used if
                               `height` is also None. Defaults to None.
        height (int, optional): The target height. If None, it's calculated from `width`
                                to maintain aspect ratio, or original height is used if
                                `width` is also None. Defaults to None.
        device (str, optional): Device to use for resizing ('cpu' or 'gpu').
                                If 'gpu' is chosen but CUDA is not available,
                                it falls back to 'cpu'. Defaults to 'cpu'.
        inter (int, optional): OpenCV interpolation method to use for resizing.
                               Defaults to cv2.INTER_AREA, which is generally good
                               for shrinking images. For enlarging, cv2.INTER_LINEAR
                               or cv2.INTER_CUBIC might be preferred.

    Returns:
        np.ndarray: The resized image.
    """
    (h, w) = image.shape[:2]

    if (height is None) and (width is None):
        return image
    if (height is None) and (width is not None):
        ratio = width / np.float32(w)
        height=np.int32(h * ratio)
    elif (width is None) and (height is not None):
        ratio = height / np.float32(h)
        width = np.int32(w * ratio)
    else:
        pass

    if device=='gpu':
        if not is_cuda_cv():
            device='cpu'

    if device=='gpu':
        src = cv2.cuda_GpuMat()
        src.upload(image)
        dest = cv2.cuda.resize(src, (width,height), interpolation=inter)
        resized=dest.download()      
    else:
        resized = cv2.resize(image, (width,height), interpolation=inter)

    return resized



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Resizes images from an input directory, optionally recursively. "
                    "Preserves aspect ratio if only width or height is specified."
    )
    ap.add_argument('-i','--input_path', required=True, help='Path to the directory containing input images.')
    ap.add_argument('-o','--output_path', required=True, help='Path to the directory where resized images will be saved.')
    ap.add_argument('--width', type=int, default=None,
                        help="Target width. If not given, calculated from height or original if height also missing.")
    ap.add_argument('--height',type=int, default=None,
                        help="Target height. If not given, calculated from width or original if width also missing.")
    ap.add_argument('--recursive', action='store_true', help='If set, processes images in subdirectories as well.')
    args = ap.parse_args()

    input_path_arg = args.input_path
    output_path_arg = args.output_path
    target_height = args.height
    target_width = args.width
    recursive_flag = args.recursive
    
    if not os.path.isdir(input_path_arg):
        logger.error(f'Input path is not a directory: {input_path_arg}')
        exit(1)
    
    if target_width is None and target_height is None:
        logger.error("Please specify at least --width or --height for resizing.")
        exit(1)

    if not os.path.exists(output_path_arg):
        logger.info(f"Creating output directory: {output_path_arg}")
        os.makedirs(output_path_arg)

    # Determine preferred device (GPU if available, else CPU)
    # This could also be a CLI argument if desired.
    # For now, let's default to 'cpu' as 'gpu' might not always be faster for single images
    # and adds complexity of checking CUDA availability for each call.
    # The `resize` function itself handles fallback if 'gpu' is passed and CUDA is not there.
    resize_device = 'cpu'
    # if is_cuda_cv():
    #     logger.info("CUDA device found. Will attempt to use GPU for resizing if specified by function.")
    #     resize_device = 'gpu' # Or let the resize function default handle it unless explicitly set.
    # else:
    #     logger.info("No CUDA device found or OpenCV not built with CUDA. Using CPU for resizing.")

    relative_img_paths = get_relative_paths(input_path_arg, recursive_flag)
    if not relative_img_paths:
        logger.info(f"No images found in {input_path_arg}{' recursively' if recursive_flag else ''}.")
        exit(0)

    logger.info(f"Found {len(relative_img_paths)} images to process.")

    for rel_path in relative_img_paths:
        full_input_path = os.path.join(input_path_arg, rel_path)
        image_to_resize = cv2.imread(full_input_path)

        if image_to_resize is None:
            logger.warning(f"Could not read image: {full_input_path}. Skipping.")
            continue

        logger.info(f"Resizing {rel_path}...")
        resized_image = resize(image_to_resize, target_width, target_height, device=resize_device)

        # Determine output filename suffix based on actual output dimensions
        out_h_actual, out_w_actual = resized_image.shape[:2]

        base_filename = os.path.basename(rel_path)
        name_part, ext_part = os.path.splitext(base_filename)
        # Standardize output to PNG, or keep original extension? For now, PNG.
        output_filename_suffix = f'_resize_{out_w_actual}x{out_h_actual}.png'
        output_filename = name_part + output_filename_suffix

        output_subdir = os.path.join(output_path_arg, os.path.dirname(rel_path))
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        final_output_path = os.path.join(output_subdir, output_filename)
        try:
            cv2.imwrite(final_output_path, resized_image)
            logger.info(f'Saved resized image to: {final_output_path}')
        except Exception as e:
            logger.error(f"Error saving image {final_output_path}: {e}")
    
    logger.info("Image resizing process complete.")
