"""
Converts 16-bit grayscale profile images (height maps) into 8-bit RGB colorized
height maps using various colormapping techniques.

This script is designed to process uint16 or int16 input images, typically PNGs,
where pixel values represent height. It normalizes the height data (excluding
0, which is treated as an invalid/background value) and applies a colormap
to generate an 8-bit RGB image. This can be useful for visualizing height maps
or preparing them for models that expect RGB input.

The script supports several colormapping options:
- 'gray': Grayscale conversion.
- 'rainbow-low': Applies OpenCV's COLORMAP_JET to an 8-bit version of the data.
- 'rainbow-med': Uses a custom 'convert_array_to_rainbow' function for potentially
                 better precision with 16-bit intermediate steps.

It can be run as a command-line tool to process individual images or all compatible
images within a specified directory.

Command-line arguments:
  -i, --input_path: Path to the input image file or directory containing .png images.
                    (default: '.')
  -o, --output_path: Path to the output directory. If not specified, output files
                     are saved in the input directory. (default: None, uses input_path)
  --map_choice: Colormapping option.
                Choices: 'gray', 'rainbow-low', 'rainbow-med', 'rainbow-high'.
                (default: 'rainbow-med')
  --show_quant: If specified, prints the number of unique values in the input and
                output images to assess quantization effects (can be slow).
                (optional, default: False)

Example usage (command-line):
  # Convert a single profile image using the default rainbow-med map
  python profile_to_hmap.py -i profile_image.png -o output_hmaps

  # Convert all .png images in 'input_profiles' directory to grayscale height maps
  python profile_to_hmap.py -i input_profiles -o output_gray_hmaps --map_choice gray

  # Convert and show quantization info
  python profile_to_hmap.py -i profile.png --map_choice rainbow-low --show_quant
"""
#%%
import cv2
import numpy as np
from image_utils.rgb_converter import convert_array_to_rainbow

BLACK=[0,0,0]
TWO_TO_SIXTEEN_MINUS_ONE=np.power(2,16)-1
TWO_TO_FIFTEEN=np.power(2,15)

def img_rgb_to_int_array(img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image into an array of unique integer values representing pixel colors.

    This function is primarily intended for testing and debugging, specifically to count
    the number of unique colors in an RGB image after colormapping. Each RGB pixel
    (r, g, b) is packed into a single integer: (r << 16) | (g << 8) | b.
    If the input is already a 2D (grayscale) array, it's returned as is.

    Args:
        img (np.ndarray): The input RGB image (HxWx3) or grayscale image (HxW).

    Returns:
        np.ndarray: A 1D array of integers if input was RGB, otherwise the original
                    grayscale image array.
    """
    if len(img.shape)>2:
        arr=img.reshape(-1,3)
        arr_int=[]
        for element in arr:
            r=element[0]
            g=element[1]
            b=element[2]
            binary_word = (r << 16) | (g << 8) | b
            arr_int.append(binary_word)
        arr_int=np.array(arr_int)
    else:
        arr_int=img
    return arr_int

# Process data
def preprocess_hmap(img: np.ndarray,
                    map_choice: str = 'rainbow-med',
                    global_max: int = None,
                    remove_outliers: bool = False) -> np.ndarray:
    """
    Converts a 16-bit profile image (height map) into an 8-bit RGB colorized height map.

    The function performs several steps:
    1. Converts input int16 or uint16 image to a working uint16 representation.
       Values of 0 in the original uint16 or -32768 in int16 are treated as invalid.
    2. Normalizes the height values: shifts the valid range so the lowest valid height
       becomes 1, and 0 remains the invalid data marker.
    3. Optionally removes outliers (pixels with height > mean + 3*std_dev of valid heights).
    4. Normalizes the image to the range [0, 1] based on either its own max or `global_max`.
    5. Applies the chosen colormap ('gray', 'rainbow-low', 'rainbow-med').

    Args:
        img (np.ndarray): Input 16-bit profile image (int16 or uint16).
        map_choice (str, optional): Colormap option.
                                    'gray': Grayscale.
                                    'rainbow-low': OpenCV COLORMAP_JET on 8-bit data.
                                    'rainbow-med': Custom rainbow on 16-bit intermediate.
                                    Defaults to 'rainbow-med'.
        global_max (int, optional): If provided, this value is used as the maximum
                                    height for normalization. Pixels above this are capped.
                                    Should be relative to the original uint16 range if used.
                                    Defaults to None (use image's own max).
        remove_outliers (bool, optional): If True, removes outlier pixels before normalization.
                                          Defaults to False.

    Returns:
        np.ndarray: The 8-bit RGB colorized height map (HxWx3).

    Raises:
        Exception: If input image dtype is not int16 or uint16, or if an
                   unsupported `map_choice` is given.
    """
    if img.dtype == np.int16:
        #convert to uint16
        img=img.astype(np.int32)+TWO_TO_FIFTEEN
        img=img.astype(np.uint16)
    elif img.dtype == np.uint16:
        img=img
    else:
        raise Exception(f'Input datatype: {img.dtype} is not supported.  Please use int16 or uint16 data.')

    # Find all unique levels in the hmap
    levels=np.unique(img)
    # Fetch first valid z height
    level_1=levels[1]
    # Fetch indices of invalid height values
    empty_ind=np.where(img==0)
    # Shift imag 1 unit above the floor to maximize the full scale range
    img=img-(level_1-1)
    # Reset the floor to 0
    img[empty_ind]=0

    # Remove outliers
    if remove_outliers:
        ind_valid=np.where(img!=0)
        img_mean=img[ind_valid].mean()
        img_std=img[ind_valid].std()
        int_outlier=np.where(img>img_mean+3*img_std)
        img[int_outlier]=0

    # Normalize image
    if global_max==None:
        img_max=img.max()
    else:
        img_max=global_max-(level_1-1)
        img[img>img_max]=img_max
        
    img_n=img/img_max

    # Convert to Grayscale
    if map_choice=='gray':   
        hmap=(img_n*255.0).astype(np.uint8)

    # Convert to Rainbow Low Precision
    elif map_choice=='rainbow-low':
        img_gray=(img_n*255.0).astype(np.uint8)
        hmap=cv2.applyColorMap(img_gray,cv2.COLORMAP_JET)
        hmap=cv2.cvtColor(hmap,cv2.COLOR_BGR2RGB)
        hmap[empty_ind]=BLACK

    # %% Convert to Med precision Rainbow
    elif map_choice=='rainbow-med':
        img_fsr=(img_n*TWO_TO_SIXTEEN_MINUS_ONE).astype(np.uint16)
        hmap=convert_array_to_rainbow(img_fsr,full_scale_range=16)

    # elif map_choice=='rainbow-high':
    #     import matplotlib as plt
    #     #TODO: This doesn't seem to improve quantization
    #     colormap = plt.get_cmap('jet')
    #     img_rgba = (colormap(img_n)*255.0).astype(np.uint8)
    #     # hmap=colormap(img_n)
    #     hmap=cv2.cvtColor(img_rgba,cv2.COLOR_RGBA2RGB)
    #     hmap[empty_ind]=BLACK

    else:
        raise Exception(f'Unsupported colormapping option: {map_choice}')

    
    return hmap

if __name__=='__main__':
    import argparse
    import glob
    import os
    import time

    ap = argparse.ArgumentParser(
        description="Converts 16-bit profile images to 8-bit RGB colorized height maps."
    )
    ap.add_argument('-i','--input_path', default='.',
                        help="Path to the input image file or directory containing .png images. (default: '.')")
    ap.add_argument('-o','--output_path', default=None,
                        help="Path to the output directory. If not specified, output files are saved "
                             "in the input directory. (default: same as input_path)")
    mapping_options = ["gray", "rainbow-low", "rainbow-med", "rainbow-high"]
    ap.add_argument('--map_choice', choices=mapping_options, default='rainbow-med',
                        help="Colormapping option. Choices: 'gray', 'rainbow-low', 'rainbow-med', "
                             "'rainbow-high'. (default: 'rainbow-med')")
    ap.add_argument('--show_quant', action='store_true',
                        help="If specified, prints the number of unique values in input/output images "
                             "to assess quantization (can be slow). (default: False)")
    # global_max and remove_outliers can be added as CLI arguments if needed
    # ap.add_argument('--global_max', type=int, default=None, help="Global max value for normalization.")
    # ap.add_argument('--remove_outliers', action='store_true', help="Remove outliers before normalization.")
    
    args = ap.parse_args()

    input_path_arg = args.input_path
    output_path_arg = args.output_path
    map_choice_arg = args.map_choice
    show_quant_arg = args.show_quant

    if output_path_arg is None:
        output_path_arg = input_path_arg

    if os.path.isdir(input_path_arg):
        files_to_process = glob.glob(os.path.join(input_path_arg, '*.png'))
        if not files_to_process:
            print(f"[INFO] No .png files found in directory: {input_path_arg}")
            exit()
    elif os.path.isfile(input_path_arg):
        if not input_path_arg.lower().endswith('.png'):
            print(f"[ERROR] Input file is not a .png file: {input_path_arg}")
            exit()
        files_to_process = [input_path_arg]
    else:
        print(f"[ERROR] Input path is not a valid file or directory: {input_path_arg}")
        exit()
    
    if not os.path.exists(output_path_arg):
        print(f"[INFO] Creating output directory: {output_path_arg}")
        os.makedirs(output_path_arg)

    processing_times = []
    print(f"[INFO] Using colormap: '{map_choice_arg}'")

    for file_path in files_to_process:
        print(f"[INFO] Processing file: {file_path}")
        img_profile = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) # Read as is, including depth

        if img_profile is None:
            print(f"[WARNING] Could not read image: {file_path}. Skipping.")
            continue

        if show_quant_arg:
            unique_input_values = len(np.unique(img_profile))
            print(f'[INFO] Input image has {unique_input_values} unique values.')

        start_time = time.time()
        # For CLI, global_max and remove_outliers are not exposed yet, using defaults.
        hmap_rgb = preprocess_hmap(img_profile, map_choice=map_choice_arg, global_max=None, remove_outliers=False)
        end_time = time.time()

        time_delta = end_time - start_time
        processing_times.append(time_delta)
        print(f'[INFO] Processing time for {os.path.basename(file_path)}: {time_delta:.4f} seconds')

        if show_quant_arg:
            hmap_int_values = img_rgb_to_int_array(hmap_rgb)
            unique_output_values = len(np.unique(hmap_int_values))
            print(f'[INFO] Converted height map has {unique_output_values} unique color values.')

        base_filename = os.path.basename(file_path)
        output_filename = os.path.splitext(base_filename)[0] + f'_hmap_{map_choice_arg}.png'

        # Convert RGB (from preprocess_hmap) to BGR for OpenCV imwrite
        hmap_bgr = cv2.cvtColor(hmap_rgb, cv2.COLOR_RGB2BGR)

        try:
            cv2.imwrite(os.path.join(output_path_arg, output_filename), hmap_bgr)
            print(f"[INFO] Saved: {os.path.join(output_path_arg, output_filename)}")
        except Exception as e:
            print(f"[ERROR] Could not save image {output_filename}: {e}")


    if processing_times:
        mean_proc_time = np.mean(processing_times)
        print(f'[INFO] Mean Processing Time per image: {mean_proc_time:.4f} seconds')
    print("[INFO] All processing complete.")
        



