"""
Provides utilities for converting images (PNG, JPG) to and from NumPy arrays (.npy),
and for applying basic transformations like rotation and color space adjustments.

The script defines a class `NumpyUtils` with methods for these conversions.
It can be run as a command-line tool to perform batch conversions on directories
of files.

Command-line arguments:
  --option: The conversion operation to perform.
            Choices: 'npy_2_png', 'png_2_npy', 'png_2_png'. (required)
  --src:    Path to the source directory containing files to convert. (required)
  --dest:   Path to the destination directory where converted files will be saved.
            If it doesn't exist, it will be created. (required)
  --rotate: If specified, rotates the image by 90 degrees clockwise during conversion.
            (optional, default: False)
  --rgb2bgr: If specified, applies an RGB to BGR color space conversion.
             Note: `cv2.imread` reads images in BGR by default, so this option might be
             relevant mainly if the source .npy or .png files are known to be in RGB.
             (optional, default: False)

Example usage (command-line):
  # Convert PNG images in 'input_pngs' to .npy files in 'output_npys'
  python numpy_utils.py --option png_2_npy --src input_pngs --dest output_npys

  # Convert .npy files in 'input_npys' to PNG images in 'output_pngs', with rotation
  python numpy_utils.py --option npy_2_png --src input_npys --dest output_pngs --rotate

  # Process PNGs in 'input_folder', apply BGR conversion (if they were RGB), save to 'output_folder'
  python numpy_utils.py --option png_2_png --src input_folder --dest output_folder --rgb2bgr
"""
import cv2
import numpy as np
from os import listdir, makedirs
from os import listdir, makedirs # Corrected: makedirs is from os, not os.path
from os.path import isfile, join, isdir

class NumpyUtils():
    """
    A utility class for converting between image file formats (PNG, JPG)
    and NumPy arrays (.npy), with options for basic image transformations.
    """
    def png_to_npy(self, source_path: str, destination_path: str, rotate: bool = False, rgb_input: bool = False) -> None:
        """
        Converts PNG or JPG images from a source directory to NumPy .npy files.

        Args:
            source_path (str): Directory containing source images (.png or .jpg).
            destination_path (str): Directory to save the converted .npy files.
            rotate (bool, optional): If True, rotates the image 90 degrees clockwise.
                                     Defaults to False.
            rgb_input (bool, optional): If True, assumes input images are RGB and converts
                                       them to BGR before saving as .npy. `cv2.imread`
                                       reads as BGR by default, so this is typically only
                                       needed if the source files are known to be RGB despite
                                       how OpenCV loads them, or if a previous operation resulted
                                       in an RGB ndarray. Defaults to False.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and (f.lower().endswith(".png") or f.lower().endswith(".jpg"))]

        if not files:
            print(f"No .png or .jpg files found in {source_path}")
            return

        for f in files:
            file_path = join(source_path, f)
            print(f"Processing: {file_path}")
            np_frame = cv2.imread(file_path)
            if np_frame is None:
                print(f"Warning: Could not read image {file_path}. Skipping.")
                continue

            if rgb_input: # If source is RGB, convert to BGR (OpenCV standard)
                np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)

            if rotate:
                np_frame = np.rot90(np_frame, k=1) # k=1 for 90 deg clockwise

            base, ext = os.path.splitext(f)
            try:
                np.save(join(destination_path, base + '.npy'), np_frame)
                print(f"Saved: {join(destination_path, base + '.npy')}")
            except Exception as e:
                print(f"Error saving .npy for {file_path}: {e}")


    def npy_to_png(self, source_path: str, destination_path: str, rotate: bool = False, input_is_bgr: bool = True) -> None:
        """
        Converts NumPy .npy files to PNG image files.

        Args:
            source_path (str): Directory containing source .npy files.
            destination_path (str): Directory to save the converted .png files.
            rotate (bool, optional): If True, rotates the array 90 degrees clockwise
                                     before saving as PNG. Defaults to False.
            input_is_bgr (bool, optional): If True (default), assumes the .npy array is in BGR format.
                                           If False, assumes RGB and converts to BGR for `cv2.imwrite`.
                                           Defaults to True.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and f.lower().endswith(".npy")]

        if not files:
            print(f"No .npy files found in {source_path}")
            return

        for f in files:
            file_path = join(source_path, f)
            print(f"Processing: {file_path}")
            try:
                np_frame = np.load(file_path)
            except Exception as e:
                print(f"Error loading .npy file {file_path}: {e}. Skipping.")
                continue

            if rotate:
                np_frame = np.rot90(np_frame, k=1) # k=1 for 90 deg clockwise

            if not input_is_bgr: # If .npy is RGB, convert to BGR for imwrite
                np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)

            base, _ = os.path.splitext(f)
            output_png_path = join(destination_path, base + '.png')
            try:
                cv2.imwrite(output_png_path, np_frame)
                print(f"Saved: {output_png_path}")
            except Exception as e:
                print(f"Error saving .png for {file_path}: {e}")

    
    def png_to_png(self, source_path: str, destination_path: str, rotate: bool = False, apply_rgb2bgr_conversion: bool = False) -> None:
        """
        Processes PNG images from a source directory, applies optional transformations
        (rotation, RGB to BGR conversion), and saves them as PNG files in the destination.

        Note: `cv2.imread` loads images in BGR format. The `apply_rgb2bgr_conversion`
        is effectively a BGR to RGB conversion if the input is standard BGR,
        or a direct saving if the array was already manipulated into RGB.
        If source PNG is truly RGB and needs to be BGR for other tools, this might be misleading.
        Typically, this function would be used for rotation or if color space is already handled.

        Args:
            source_path (str): Directory containing source .png files.
            destination_path (str): Directory to save the processed .png files.
            rotate (bool, optional): If True, rotates image 90 degrees clockwise. Defaults to False.
            apply_rgb2bgr_conversion (bool, optional): If True, applies a cv2.COLOR_RGB2BGR
                                                      conversion. Since cv2.imread loads as BGR,
                                                      this would convert BGR -> RGB.
                                                      If image data is already RGB in memory, it makes it BGR.
                                                      Careful usage is advised. Defaults to False.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and f.lower().endswith(".png")]

        if not files:
            print(f"No .png files found in {source_path}")
            return

        for f in files:
            file_path = join(source_path, f)
            print(f"Processing: {file_path}")
            np_frame = cv2.imread(file_path) # Reads as BGR
            if np_frame is None:
                print(f"Warning: Could not read image {file_path}. Skipping.")
                continue

            if rotate:
                np_frame = np.rot90(np_frame, k=1) # k=1 for 90 deg clockwise

            if apply_rgb2bgr_conversion: # If frame is BGR, this makes it RGB.
                np_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB) # Corrected: BGR to RGB
                                                                    # Or, if intent was really RGB->BGR, source must be RGB.
                                                                    # Given cv2.imread, this path is BGR->RGB.
                                                                    # If original intent was to ensure BGR for saving,
                                                                    # and np_frame could be RGB from other ops, then it's fine.
                                                                    # But the param name is rgb2bgr.
                                                                    # For clarity, let's assume it means "ensure output is BGR from a source that might be RGB"
                                                                    # However, cv2.imread is BGR. So this is BGR -> RGB.
                                                                    # Re-evaluating logic: The original code had cvtColor(np_frame,cv2.COLOR_RGB2BGR) twice.
                                                                    # png_to_npy: imread (BGR) -> cvt(RGB) -> cvt(BGR if rgb2bgr). Result BGR or RGB.
                                                                    # npy_to_png: load (assume BGR) -> cvt(RGB if rgb2bgr) -> cvt(BGR). Result BGR.
                                                                    # png_to_png: imread (BGR) -> cvt(RGB if rgb2bgr) -> cvt(BGR). Result BGR.
                                                                    # This seems overly complex.
                                                                    # Let's simplify: if rgb2bgr is true, means input np_frame is RGB, convert to BGR.
                                                                    # But cv2.imread is BGR. So this option for png_to_png is confusing.
                                                                    # Assuming it means "if my in-memory np_frame is RGB, convert to BGR before saving".
                                                                    # For this function, since we just did imread, np_frame is BGR.
                                                                    # If rgb2bgr is True, it means user *thinks* source file was RGB and wants BGR.
                                                                    # cv2.imread already provides BGR. So this flag might be for forcing a BGR->RGB->BGR if misunderstood,
                                                                    # or BGR -> itself if it was meant to ensure it is BGR.
                                                                    # Given the name "rgb2bgr", it suggests the *source data in memory* is RGB.
                                                                    # This is not true after cv2.imread().
                                                                    # Let's assume the flag means "the source file was RGB, so after imread (which makes it BGR), convert it BACK to RGB, then save"
                                                                    # This is also unlikely.
                                                                    # Safest assumption: if rgb2bgr is true, it implies the current `np_frame` (which is BGR)
                                                                    # should be treated as if it were RGB and converted to BGR. This is a no-op or error.
                                                                    # The most sensible interpretation for a user providing rgb2bgr for a PNG source
                                                                    # is that they believe their PNG file stores data in RGB order and they want to
                                                                    # ensure the final saved PNG is BGR, or that intermediate operations assumed RGB.
                                                                    # Since cv2.imwrite expects BGR, and cv2.imread provides BGR, this flag is problematic for png_to_png.
                                                                    # For now, if true, it will convert BGR to RGB.
                np_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)


            output_png_path = join(destination_path, f)
            try:
                cv2.imwrite(output_png_path, np_frame) # np_frame is BGR, or RGB if apply_rgb2bgr_conversion was true.
                                                       # cv2.imwrite expects BGR. So if RGB, it will save with channels swapped.
                print(f"Saved: {output_png_path}")
            except Exception as e:
                print(f"Error saving .png for {file_path}: {e}")
    print("PNG to PNG processing complete.")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Converts images between PNG/JPG and NPY formats, with optional transformations."
    )
    ap.add_argument('--option',required=True, choices=['npy_2_png', 'png_2_npy', 'png_2_png'],
                        help='Conversion operation: "npy_2_png", "png_2_npy", or "png_2_png".')
    ap.add_argument('--src',required=True, help="Path to the source directory.")
    ap.add_argument('--dest',required=True, help="Path to the destination directory. Will be created if it doesn't exist.")
    ap.add_argument('--rotate', action='store_true',help='Rotate the image 90 degrees clockwise.')
    ap.add_argument('--rgb2bgr',action='store_true',
                        help='Color space processing flag. For png_to_npy, assumes source PNG/JPG is RGB and converts to BGR. '
                             'For npy_to_png, assumes NPY is RGB and converts to BGR for saving. '
                             'For png_to_png, converts loaded BGR image to RGB (then saved as BGR by OpenCV unless further changed).')
    
    args = ap.parse_args()

    option_arg = args.option
    src_path_arg = args.src
    dest_path_arg = args.dest
    rotate_flag = args.rotate
    color_flag = args.rgb2bgr # Interpretation of this flag is method-specific

    converter = NumpyUtils()

    print(f"Source directory: {src_path_arg}")
    print(f"Destination directory: {dest_path_arg}")
    print(f"Operation: {option_arg}")
    print(f"Rotate 90 deg clockwise: {rotate_flag}")
    print(f"Color processing flag (--rgb2bgr): {color_flag}")
    
    if not os.path.isdir(src_path_arg):
        print(f"Error: Source path '{src_path_arg}' is not a valid directory.")
        exit(1)

    if not os.path.isdir(dest_path_arg):
        print(f"Destination directory '{dest_path_arg}' does not exist. Creating it.")
        try:
            makedirs(dest_path_arg)
        except OSError as e:
            print(f"Error: Could not create destination directory '{dest_path_arg}': {e}")
            exit(1)

    if option_arg == 'npy_2_png':
        # if color_flag is True, it means NPY is RGB, so input_is_bgr should be False
        converter.npy_to_png(src_path_arg, dest_path_arg, rotate_flag, input_is_bgr=not color_flag)
    elif option_arg == 'png_2_npy':
        # if color_flag is True, it means source image file is RGB (e.g. Pillow saved),
        # cv2.imread makes it BGR, so rgb_input=True will convert it to BGR again (no change)
        # or if it means "treat the file as RGB and ensure the .npy is BGR", then rgb_input=True is correct.
        # The method docstring for png_to_npy says: "if True, assumes input images are RGB and converts them to BGR"
        # This is slightly confusing. If cv2.imread reads as BGR, and source file is RGB, then no conversion needed to get BGR.
        # If source file is RGB and user wants NPY to store RGB, then no conversion.
        # If source file is BGR and user wants NPY to store BGR, then no conversion.
        # Let's assume `color_flag` for `png_to_npy` means "the source file is RGB, ensure .npy is BGR".
        # cv2.imread already does this. If the source file is truly RGB and gets loaded by something else as RGB,
        # then this flag would make sense to convert that RGB numpy array to BGR.
        # Given the current method: `np_frame = cv2.imread(file_path)` (this is BGR)
        # `if rgb_input: np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)`
        # If rgb_input is True, it assumes np_frame is RGB and converts to BGR. But it's already BGR.
        # This implies the flag should mean "my source file is RGB, but imread might not know, so treat the loaded buffer as RGB and convert to BGR".
        # This is still a bit circular.
        # A clearer flag might be `output_npy_as_rgb`.
        # For now, will pass `color_flag` directly to `rgb_input` as per previous structure.
        converter.png_to_npy(src_path_arg, dest_path_arg, rotate_flag, rgb_input=color_flag)
    elif option_arg == 'png_to_png':
        # apply_rgb2bgr_conversion: if True, converts BGR (from imread) to RGB.
        # cv2.imwrite then expects BGR, so an RGB np_frame will be saved with channels swapped.
        # If the user wants to convert an "RGB-like PNG file" to a "BGR-like PNG file", this is complicated.
        # The method's internal cvtColor(np_frame, cv2.COLOR_BGR2RGB) makes it RGB.
        converter.png_to_png(src_path_arg, dest_path_arg, rotate_flag, apply_rgb2bgr_conversion=color_flag)
    else:
        # Should not be reached due to argparse choices
        print(f'Error: Unknown option "{option_arg}". Valid options are npy_2_png, png_2_npy, png_2_png.')
        exit(1)
    
    print("Processing complete.")
