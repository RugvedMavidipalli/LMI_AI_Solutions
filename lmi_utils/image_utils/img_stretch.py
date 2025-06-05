"""
Stretches or compresses images based on their aspect ratio and a stretch factor.

This script processes PNG images from an input path (either a single file or
a directory of files).
- If an image's width is greater than its height, its width is compressed by
  the `wh_stretch` factor.
- If an image's height is greater than or equal to its width, its height is
  stretched by the `wh_stretch` factor.

The output images are saved in the specified output directory with '_stretch.png'
appended to their original filenames.

Command-line arguments:
  -i, --input_path: Path to the input image file (.png) or directory containing .png images.
                    (default: '.')
  -o, --output_path: Path to the directory where stretched images will be saved.
                     If it doesn't exist, it will be created. (default: './stretch')
  --wh_stretch: The factor by which to stretch or compress the image dimension.
                (default: 3.0)

Example usage (command-line):
  # Stretch images in 'input_dir', save to 'output_stretched_dir' with stretch factor 2.5
  python img_stretch.py -i input_dir -o output_stretched_dir --wh_stretch 2.5

  # Stretch a single image 'image.png' with default settings
  python img_stretch.py -i image.png
"""
import argparse
import glob
import os
import cv2
import numpy as np # Added for np.ndarray type hint

def stretch(img: np.ndarray, wh_stretch_factor: float) -> np.ndarray:
    """
    Stretches or compresses an image based on its aspect ratio and a stretch factor.

    - If width > height, the width is divided by `wh_stretch_factor` (compressed).
    - If height >= width, the height is multiplied by `wh_stretch_factor` (stretched).
    Resizing is done using OpenCV's `cv2.resize` with `cv2.INTER_AREA` interpolation.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        wh_stretch_factor (float): The factor to apply for stretching or compressing.
                                   Must be a positive number. If 0 or negative,
                                   original image is returned.

    Returns:
        np.ndarray: The stretched or compressed image.
    """
    if wh_stretch_factor <= 0:
        print("[WARNING] wh_stretch_factor must be positive. Returning original image.")
        return img

    h,w=img.shape[:2]
    if w>h: # Compress width
        new_w=int(w / wh_stretch_factor)
        if new_w == 0: # Avoid zero dimension
            print("[WARNING] Calculated new width is 0. Check wh_stretch_factor. Returning original image.")
            return img
        print(f'[INFO] Compressing width: original w={w} to new_w={new_w}. Aspect ratio preserved for height.')
        dim=(new_w,h)
        img_stretch=cv2.resize(img,dim,cv2.INTER_AREA)
    else: # Stretch height
        new_h=int(h * wh_stretch_factor)
        if new_h == 0: # Avoid zero dimension
            print("[WARNING] Calculated new height is 0. Check wh_stretch_factor. Returning original image.")
            return img
        print(f'[INFO] Stretching height: original h={h} to new_h={new_h}. Aspect ratio preserved for width.')
        dim=(w,new_h)
        img_stretch=cv2.resize(img,dim,cv2.INTER_AREA)
    return img_stretch

if __name__=="__main__":
    ap = argparse.ArgumentParser(
        description="Stretches or compresses PNG images based on aspect ratio and a stretch factor."
    )
    ap.add_argument('-i','--input_path',default='.',
                        help="Path to the input image file (.png) or directory containing .png images. (default: '.')")
    ap.add_argument('-o','--output_path',default='./stretch',
                        help="Path to the directory where stretched images will be saved. (default: './stretch')")
    ap.add_argument('--wh_stretch',type=float,default=3.0,
                        help="The factor by which to stretch or compress. (default: 3.0)")
    args = ap.parse_args()

    input_path_arg = args.input_path
    output_path_arg = args.output_path
    wh_stretch_factor_arg = args.wh_stretch

    if wh_stretch_factor_arg <= 0:
        print("[ERROR] --wh_stretch factor must be a positive number.")
        exit(1)

    files_to_process = []
    if os.path.isdir(input_path_arg):
        files_to_process = glob.glob(os.path.join(input_path_arg, '*.png'))
        if not files_to_process:
            print(f"[INFO] No .png files found in directory: {input_path_arg}")
            exit(0)
    elif os.path.isfile(input_path_arg):
        if input_path_arg.lower().endswith('.png'):
            files_to_process = [input_path_arg]
        else:
            print(f"[ERROR] Input file is not a .png file: {input_path_arg}")
            exit(1)
    else:
        print(f"[ERROR] Input path is not a valid file or directory: {input_path_arg}")
        exit(1)

    if not os.path.exists(output_path_arg):
        print(f"[INFO] Creating output directory: {output_path_arg}")
        os.makedirs(output_path_arg)

    print(f"[INFO] Processing {len(files_to_process)} image(s)...")
    for file_path in files_to_process:
        img_original = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) # Keep original depth, e.g., for 16-bit images

        if img_original is None:
            print(f"[WARNING] Could not read image: {file_path}. Skipping.")
            continue

        print(f"[INFO] Processing: {file_path}")
        img_stretched = stretch(img_original, wh_stretch_factor_arg)

        base_filename = os.path.basename(file_path)
        name_part, ext_part = os.path.splitext(base_filename)
        output_filename = name_part + '_stretch' + ext_part # Keep original extension

        output_file_path = os.path.join(output_path_arg, output_filename)
        try:
            cv2.imwrite(output_file_path, img_stretched)
            print(f"[INFO] Saved stretched image to: {output_file_path}")
        except Exception as e:
            print(f"[ERROR] Could not save image {output_file_path}: {e}")

    print("[INFO] Image stretching process complete.")


