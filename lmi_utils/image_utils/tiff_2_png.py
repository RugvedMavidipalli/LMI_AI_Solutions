"""
Converts TIFF (.tiff or .tif) image files to PNG (.png) format.

This script processes TIFF images from a specified input path (either a single
file or a directory of .tiff/.tif files). It can optionally apply a JET colormap
to the images during conversion, which is useful for visualizing single-channel
(e.g., grayscale, height map) TIFF images as colorized PNGs.

Command-line arguments:
  -i, --input_path: Path to the input TIFF image file or directory containing
                    .tiff/.tif images. (default: '.')
  -o, --output_path: Path to the directory where converted PNG images will be saved.
                     If it doesn't exist, it will be created. (default: './png')
  --cvt_hmap_jet: If specified, applies cv2.COLORMAP_JET to the image before saving.
                  This is useful for visualizing single-channel height maps.
                  (optional, default: False)

Example usage (command-line):
  # Convert all .tiff files in 'input_tiffs' to .png files in 'output_pngs'
  python tiff_2_png.py -i input_tiffs -o output_pngs

  # Convert a single TIFF image and apply JET colormap
  python tiff_2_png.py -i my_image.tiff -o output_images --cvt_hmap_jet
"""
import argparse
import glob
import os
import cv2
from typing import List

def convert_tiff_to_png(input_path: str, output_path: str, apply_colormap: bool = False) -> None:
    """
    Converts TIFF image(s) from `input_path` to PNG format and saves them in `output_path`.

    Args:
        input_path (str): Path to a single .tiff image file or a directory containing .tiff files.
        output_path (str): Directory where the converted .png files will be saved.
        apply_colormap (bool, optional): If True, applies `cv2.COLORMAP_JET` to the image
                                         before saving. This is typically used for visualizing
                                         single-channel images like height maps. Defaults to False.

    Raises:
        FileNotFoundError: If `input_path` is not a valid file or directory.
        Exception: For other issues like unsupported file types if a single file is passed
                   that is not a .tiff/.tif.
    """
    files_to_process: List[str] = []
    if os.path.isdir(input_path):
        # Support both .tiff and .tif extensions
        for ext in ['*.tiff', '*.tif']:
            files_to_process.extend(glob.glob(os.path.join(input_path, ext)))
        if not files_to_process:
            print(f"[INFO] No .tiff or .tif files found in directory: {input_path}")
            return
    elif os.path.isfile(input_path):
        if not input_path.lower().endswith(('.tiff', '.tif')):
            raise ValueError(f"Input file is not a .tiff or .tif file: {input_path}")
        files_to_process = [input_path]
    else:
        raise FileNotFoundError(f"Input path is not a valid file or directory: {input_path}")

    if not os.path.exists(output_path):
        print(f"[INFO] Creating output directory: {output_path}")
        os.makedirs(output_path)

    print(f"[INFO] Found {len(files_to_process)} image(s) to convert.")
    for file_path in files_to_process:
        print(f'[INFO] Converting: {file_path}')
        # cv2.IMREAD_UNCHANGED is important for reading TIFFs that might have > 8-bit depth or alpha
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARNING] Could not read TIFF image: {file_path}. Skipping.")
            continue

        if apply_colormap:
            # COLORMAP_JET expects an 8-bit single-channel or 3-channel BGR image.
            # If img is, for example, 16-bit, it needs normalization to 8-bit first.
            if img.dtype == np.uint16:
                print("[INFO] Image is uint16, normalizing to uint8 for colormap application.")
                img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if len(img.shape) == 2: # Grayscale
                     img_colormap_input = img_normalized
                elif len(img.shape) == 3 and img.shape[2] >=3: # Multi-channel, take first channel or convert to gray
                     print("[INFO] Multi-channel uint16 image, converting to grayscale for colormap.")
                     img_colormap_input = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY) # Assuming BGR(A) like structure
                else: # Other cases
                    img_colormap_input = img_normalized
            elif img.dtype == np.uint8:
                if len(img.shape) == 3 and img.shape[2] >= 3: # e.g. BGR or BGRA
                    # applyColorMap works on BGR. If it's BGRA, it might also work or need conversion.
                    # For simplicity, assume BGR or single channel is what user wants for colormap.
                    # If it's a 3-channel uint8, applyColorMap will work directly.
                    img_colormap_input = img
                elif len(img.shape) == 2: # Grayscale uint8
                    img_colormap_input = img
                else:
                    print(f"[WARNING] Image {os.path.basename(file_path)} has an unhandled uint8 format for colormap: {img.shape}. Skipping colormap.")
                    img_colormap_input = img # Fallback to original image for saving
            else: # Other dtypes like float32
                print(f"[WARNING] Image {os.path.basename(file_path)} has dtype {img.dtype}. "
                      "Attempting normalization for colormap. Results may vary.")
                # General normalization for other types
                img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if len(img_normalized.shape) == 2:
                    img_colormap_input = img_normalized
                elif len(img_normalized.shape) == 3 and img_normalized.shape[2] >=3:
                     img_colormap_input = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)
                else:
                    img_colormap_input = img_normalized

            # Apply colormap only if img_colormap_input is single channel, as expected by some colormaps
            if len(img_colormap_input.shape) == 2 or (len(img_colormap_input.shape) == 3 and img_colormap_input.shape[2] == 1):
                 img_to_save = cv2.applyColorMap(img_colormap_input, cv2.COLORMAP_JET)
            elif len(img_colormap_input.shape) == 3 and img_colormap_input.shape[2] ==3: # if it's already BGR uint8
                 img_to_save = cv2.applyColorMap(img_colormap_input, cv2.COLORMAP_JET) # this will apply on BGR
            else:
                 print(f"[WARNING] Colormap input for {os.path.basename(file_path)} is not single channel or BGR uint8 after normalization. Saving without colormap.")
                 img_to_save = img # Fallback to original image if colormap input is not suitable
        else:
            img_to_save = img

        base_filename = os.path.basename(file_path)
        output_filename_base = os.path.splitext(base_filename)[0]
        output_filename = output_filename_base + '.png'

        output_file_path = os.path.join(output_path, output_filename)
        try:
            cv2.imwrite(output_file_path, img_to_save)
            print(f"[INFO] Saved: {output_file_path}")
        except Exception as e:
            print(f"[ERROR] Could not save PNG image {output_file_path}: {e}")
    print("[INFO] TIFF to PNG conversion process complete.")

if __name__=="__main__":
    import numpy as np # Import numpy here for the colormap dtype checks
    ap = argparse.ArgumentParser(
        description="Converts TIFF (.tiff, .tif) images to PNG (.png) format, optionally applying JET colormap."
    )
    ap.add_argument('-i','--input_path',default='.',
                        help="Path to the input TIFF image file or directory. (default: '.')")
    ap.add_argument('-o','--output_path',default='./png',
                        help="Path to the output directory for PNG files. (default: './png')")
    ap.add_argument('--cvt_hmap_jet', action='store_true',
                        help='If set, applies cv2.COLORMAP_JET to the image (useful for height maps).')
    args = ap.parse_args()

    input_path_arg = args.input_path
    output_path_arg = args.output_path
    apply_colormap_flag = args.cvt_hmap_jet

    try:
        convert_tiff_to_png(input_path_arg, output_path_arg, apply_colormap_flag)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


