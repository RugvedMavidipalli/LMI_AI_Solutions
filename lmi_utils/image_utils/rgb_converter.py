"""
integer converter: converts between integers and rgb encodings [red,green,blue]
image converter: converts images
"""

import numpy as np
from typing import Tuple, Union # For type hinting

TWO_TO_TWENTYFORUTH_MINUS_ONE = 16777215  # 2^24 - 1
TWO_TO_SIXTEENTH_MINUS_ONE = 65535      # 2^16 - 1

def convert_to_rgb(an_int: int) -> Tuple[np.uint8, np.uint8, np.uint8]:
    """
    Converts a 24-bit integer into its corresponding (Red, Green, Blue) color components.

    The integer is unpacked such that:
    - Red is the most significant 8 bits.
    - Green is the middle 8 bits.
    - Blue is the least significant 8 bits.

    Args:
        an_int (int): A 24-bit integer representing an RGB color.
                      Values outside [0, 16777215] will behave according to bitwise operations.

    Returns:
        Tuple[np.uint8, np.uint8, np.uint8]: A tuple (red, green, blue) where each
                                             component is a uint8 value (0-255).
    """
    blue = an_int & 255
    green = (an_int >> 8) & 255
    red = (an_int >> 16) & 255
    return np.uint8(red), np.uint8(green), np.uint8(blue)

def convert_array_to_rgb(an_array_of_ints: np.ndarray) -> np.ndarray:
    """
    Converts a NumPy array of 24-bit integers into an RGB image array.

    Each integer in the input array is converted to its (Red, Green, Blue) components.
    The output is a 3D NumPy array (height x width x 3 channels).

    Args:
        an_array_of_ints (np.ndarray): A NumPy array where each element is a 24-bit integer
                                       representing an RGB color. Typically a 1D or 2D array.

    Returns:
        np.ndarray: An RGB image as a NumPy array of shape (..., 3) with dtype uint8.
                    The initial dimensions of `an_array_of_ints` are preserved.
    """
    # Ensure input is a NumPy array
    if not isinstance(an_array_of_ints, np.ndarray):
        an_array_of_ints = np.array(an_array_of_ints)

    blue = an_array_of_ints & 255
    green = (an_array_of_ints >> 8) & 255
    red = (an_array_of_ints >> 16) & 255
    # Stack along a new last dimension and ensure correct shape for dstack
    # If input is 1D (e.g. shape (N,)), dstack needs (N,1) or similar to create (N,1,3) then reshape.
    # However, if input is already 2D (H,W), dstack works as expected.
    # Assuming an_array_of_ints is likely HxW (an image of integers) or a flat list of pixel integers.
    # For a flat list, the output shape of dstack might not be directly image-like.
    # Let's assume input is suitable for direct dstack to HxWx3 or similar.
    rgb_image = np.dstack((red, green, blue)).astype(np.uint8)
    # If the input was 1D, dstack creates (1, N, 3). We might want (N, 3).
    if an_array_of_ints.ndim == 1 and rgb_image.shape[0] == 1 :
        rgb_image = rgb_image.reshape(-1, 3)
    elif an_array_of_ints.ndim > 0 and rgb_image.shape[:-1] != an_array_of_ints.shape:
        # If shape mismatch (e.g. dstack added an extra dim for 1D input), reshape based on original.
        rgb_image = rgb_image.reshape(list(an_array_of_ints.shape) + [3])
    return rgb_image

def convert_to_rainbow(an_int: int, full_scale_range: int = 24) -> Tuple[np.uint8, np.uint8, np.uint8]:
    """
    Converts an integer value to an RGB color using a custom rainbow colormap.

    The colormap transitions through 7 bins:
    Black -> Blue -> Cyan -> Green -> Yellow -> Red -> Magenta -> White.
    The input integer is mapped to a color based on its position within the
    `full_scale_range` (either 16-bit or 24-bit).

    Args:
        an_int (int): The input integer value.
        full_scale_range (int, optional): The bit depth for the full scale of the rainbow.
                                          Supported values are 16 (for 0-65535 range) or
                                          24 (for 0-16777215 range). Defaults to 24.

    Returns:
        Tuple[np.uint8, np.uint8, np.uint8]: (red, green, blue) uint8 color components.
                                             Returns (0,0,0) black for values outside
                                             the defined bins if an error occurs during calculation,
                                             though the try-except is broad.

    Raises:
        Exception: If `full_scale_range` is not 16 or 24.
    """
    if full_scale_range == 16:
        color_bin = TWO_TO_SIXTEENTH_MINUS_ONE // 7
    elif full_scale_range == 24:
        color_bin = TWO_TO_TWENTYFORUTH_MINUS_ONE // 7
    else:
        raise ValueError(f'Unsupported full_scale_range {full_scale_range}. Choose 16 or 24.') # Changed to ValueError
    
    # Ensure color_bin is not zero to prevent DivisionByZeroError if constants are small
    if color_bin == 0:
        # This case should ideally not be reached if constants are large enough.
        # It implies the full_scale_range is too small for 7 bins.
        return np.uint8(0), np.uint8(0), np.uint8(0)

    slope = 255.0 / color_bin # Use float division
    red, green, blue = 0, 0, 0 # Default to black

    try: 
        # from black...ramp-up blue in first bin
        if an_int < color_bin:
            x=an_int
            blue=x*slope
            green=0
            red=0
        # ramp-up green in second bin
        elif an_int>=color_bin and an_int<color_bin*2:
            x=an_int-color_bin
            blue=255
            green=x*slope
            red=0
        # ramp-down blue in third bin
        elif an_int>=color_bin*2 and an_int<color_bin*3:
            x=an_int-2*color_bin
            blue=255-x*slope
            green=255
            red=0
        # ramp-up red in the fourth bin
        elif an_int>=color_bin*3 and an_int<color_bin*4:
            x=an_int-3*color_bin
            blue=0
            green=255
            red=x*slope
        #ramp-down green in 5th bin
        elif an_int>=color_bin*4 and an_int<color_bin*5:
            x=an_int-4*color_bin
            blue=0
            green=255-x*slope
            red=255
        #ramp-up blue in 6th bin
        elif an_int>=color_bin*5 and an_int<color_bin*6:
            x=an_int-5*color_bin
            blue=x*slope
            green=0
            red=255
        #ramp-up green (white=highest)
        elif an_int>=color_bin*6 and an_int<color_bin*7:
            x=an_int-6*color_bin
            blue = 255
            green = x * slope
            red = 255
        else: # Value might be >= color_bin*7, clamp to white or last color.
              # Or if an_int is negative (though input is typically unsigned for this logic)
            if an_int >= color_bin * 7:
                 blue, green, red = 255, 255, 255 # White for values at or above the top
            # else: an_int is negative, color remains black (0,0,0)

    except Exception as e: # Catch specific exceptions if possible
        print(f'Error during rainbow conversion for value {an_int}: {e}. Returning black.')
        return np.uint8(0), np.uint8(0), np.uint8(0)

    return np.uint8(red), np.uint8(green), np.uint8(blue)

def convert_array_to_rainbow(an_array_of_ints: np.ndarray, full_scale_range: int = 24) -> np.ndarray:
    """
    Converts a NumPy array of integers into an RGB image using a custom rainbow colormap.

    Each integer in the input array is mapped to an RGB color based on its value
    within the specified `full_scale_range` (16-bit or 24-bit).
    The colormap transitions through 7 bins:
    Black -> Blue -> Cyan -> Green -> Yellow -> Red -> Magenta -> White.

    Args:
        an_array_of_ints (np.ndarray): A NumPy array of integers. Can be any shape.
        full_scale_range (int, optional): The bit depth for the full scale of the rainbow.
                                          Supported values are 16 or 24. Defaults to 24.

    Returns:
        np.ndarray: An RGB image as a NumPy array of shape (*an_array_of_ints.shape, 3)
                    with dtype uint8.

    Raises:
        ValueError: If `full_scale_range` is not 16 or 24.
        Exception: If an unexpected error occurs during processing (e.g. due to input array properties).
    """
    if not isinstance(an_array_of_ints, np.ndarray):
        an_array_of_ints = np.array(an_array_of_ints)

    if full_scale_range == 16:
        color_bin = TWO_TO_SIXTEENTH_MINUS_ONE // 7
    elif full_scale_range == 24:
        color_bin = TWO_TO_TWENTYFORUTH_MINUS_ONE // 7
    else:
        raise ValueError(f'Unsupported full_scale_range {full_scale_range}. Choose 16 or 24.')

    if color_bin == 0: # Should not happen with current constants
        # Create a black image of the same shape as input + 3 color channels
        return np.zeros(list(an_array_of_ints.shape) + [3], dtype=np.uint8)

    slope = 255.0 / color_bin # Use float division

    # Initialize R, G, B channels with the same shape as the input array
    blue = np.zeros_like(an_array_of_ints, dtype=np.float64)
    green = np.zeros_like(an_array_of_ints, dtype=np.float64)
    red = np.zeros_like(an_array_of_ints, dtype=np.float64)

    try:
        # from black...ramp-up blue in first bin
        bin1_index = an_array_of_ints < color_bin
        blue[bin1_index]=an_array_of_ints[bin1_index]*slope
        # ramp-up green in second bin
        bin2_index=np.logical_and(an_array_of_ints>=color_bin,an_array_of_ints<color_bin*2)
        blue[bin2_index]=255
        green[bin2_index]=(an_array_of_ints[bin2_index]-color_bin)*slope
        # ramp-down blue in third bin
        bin3_index=np.logical_and(an_array_of_ints>=color_bin*2,an_array_of_ints<color_bin*3)
        blue[bin3_index]=255-(an_array_of_ints[bin3_index]-2*color_bin)*slope
        green[bin3_index]=255
        red[bin3_index]=0
        # ramp-up red in the fourth bin
        bin4_index=np.logical_and(an_array_of_ints>=color_bin*3,an_array_of_ints<color_bin*4)
        blue[bin4_index]=0
        green[bin4_index]=255
        red[bin4_index]=(an_array_of_ints[bin4_index]-3*color_bin)*slope
        #ramp-down green in 5th bin
        bin5_index=np.logical_and( an_array_of_ints>=color_bin*4,an_array_of_ints<color_bin*5)
        blue[bin5_index]=0
        green[bin5_index]=255-(an_array_of_ints[bin5_index]-4*color_bin)*slope
        red[bin5_index]=255
        #ramp-up blue in 6th bin
        bin6_index=np.logical_and(an_array_of_ints>=color_bin*5,an_array_of_ints<color_bin*6)
        blue[bin6_index]=(an_array_of_ints[bin6_index]-5*color_bin)*slope
        green[bin6_index]=0
        red[bin6_index]=255
        #ramp-up green (white=highest)
        bin7_index=np.logical_and(an_array_of_ints>=color_bin*6,an_array_of_ints<color_bin*7)
            blue[bin7_index] = 255.0
            green[bin7_index] = (an_array_of_ints[bin7_index] - color_bin * 6) * slope
            red[bin7_index] = 255.0

        # For values >= color_bin*7, clamp to white
        bin_over_index = an_array_of_ints >= color_bin * 7
        blue[bin_over_index] = 255.0
        green[bin_over_index] = 255.0
        red[bin_over_index] = 255.0

    except Exception as e: # Broad exception, consider more specific ones if identifiable
        raise Exception(f'Error during array to rainbow conversion: {e}')

    # Clip values to be within [0, 255] before converting to uint8
    # This handles any potential floating point inaccuracies leading to values slightly outside range
    red = np.clip(red, 0, 255)
    green = np.clip(green, 0, 255)
    blue = np.clip(blue, 0, 255)

    rgb_image = np.dstack((red, green, blue)).astype(np.uint8)
    # Ensure output shape matches input shape + color channel
    if rgb_image.shape[:-1] != an_array_of_ints.shape:
         rgb_image = rgb_image.reshape(list(an_array_of_ints.shape) + [3])
    return rgb_image


def convert_from_rgb(rgb: Tuple[int, int, int] | np.ndarray) -> int:
    """
    Converts an RGB color tuple or array [R, G, B] into a single 24-bit integer.

    The components are packed such that Red is the most significant 8 bits,
    Green is the middle, and Blue is the least significant.

    Args:
        rgb (Tuple[int, int, int] | np.ndarray): A tuple or NumPy array
                                                 representing the (Red, Green, Blue)
                                                 color components, typically uint8.

    Returns:
        int: A 24-bit integer representation of the RGB color.
    """
    # Ensure components are integers if they are not already (e.g. from np.uint8)
    red = int(rgb[0])
    green = int(rgb[1])
    blue = int(rgb[2])
    an_int = (red << 16) + (green << 8) + blue
    return an_int


def convert_greyscale_image_to_color(greyscale_img: np.ndarray) -> np.ndarray:
    """
    Converts a grayscale image to a 24-bit RGB color image by mapping grayscale
    intensity values to a range of 2^24 colors.

    The grayscale image's min and max values are mapped to the full 24-bit color
    range (0 to 16777215). Each interpolated integer value is then converted
    to an RGB triplet. This provides a form of false coloring based on intensity.
    The output image is in channel-first format (3, H, W).

    Args:
        greyscale_img (np.ndarray): Input grayscale image as a 2D NumPy array.
                                    If the input is already a color image (3D array),
                                    it's returned as is.

    Returns:
        np.ndarray: A 3-channel RGB image (uint8) with shape (3, H, W).
    """
    if not isinstance(greyscale_img, np.ndarray):
        greyscale_img = np.array(greyscale_img)

    if len(greyscale_img.shape) > 2: # Already has channels
        # Assuming if it has >2 dims, it's already HxWxChannels or ChannelsxHxW
        # This function expects to output ChannelsxHxW, so if input is HxWxC, transpose.
        if greyscale_img.shape[0] !=3 and greyscale_img.shape[-1] == 3: # HxWxC
            return np.transpose(greyscale_img, (2,0,1))
        return greyscale_img # Assume already correct or not processable by this func

    if greyscale_img.size == 0: # Handle empty image
        return np.zeros((3, 0, 0), dtype=np.uint8)

    min_val = greyscale_img.min()
    max_val = greyscale_img.max()

    if min_val == max_val: # Uniform image, avoid division by zero in interp
        # Map to a mid-gray or a specific color if desired
        # Here, mapping to the integer representation of mid-gray (128,128,128)
        # Or simply use the single value if it maps to a specific color.
        # For simplicity, let's map the uniform value.
        # This part needs careful thought on what a "color" representation of a uniform gray should be.
        # The original interpolation would map everything to greyscale_y[0] if min=max.
        # Let's use the original logic's behavior for consistency for now.
        # The interpolation will map the single value to the start of greyscale_y.
        pass

    # Define the target 24-bit integer range
    greyscale_y_target_range = np.arange(0, TWO_TO_TWENTYFORUTH_MINUS_ONE, dtype=np.int32)
    # Define the source grayscale value range
    greyscale_x_source_range = np.linspace(min_val, max_val, num=TWO_TO_TWENTYFORUTH_MINUS_ONE, dtype=greyscale_img.dtype)

    # Interpolate pixel values to the 24-bit integer range
    # Using np.int32 for greyscale_int to avoid overflow before converting to RGB components
    greyscale_int = np.interp(greyscale_img, greyscale_x_source_range, greyscale_y_target_range).astype(np.int32)

    num_rows, num_cols = greyscale_img.shape
    color_img = np.zeros((3, num_rows, num_cols), dtype=np.uint8) # Channel-first format

    for i in range(num_rows):
        for j in range(num_cols):
            r, g, b = convert_to_rgb(greyscale_int[i, j])
            color_img[0, i, j] = r
            color_img[1, i, j] = g
            color_img[2, i, j] = b
    return color_img


def convert_greyscale_to_color_simple(greyscale_img: np.ndarray) -> np.ndarray:
    """
    Converts a grayscale image to a 3-channel RGB image by replicating the
    grayscale channel three times.

    Args:
        greyscale_img (np.ndarray): Input grayscale image as a 2D NumPy array.
                                    If the input is already a color image (3D array),
                                    it's returned as is (or transposed if HxWxC).

    Returns:
        np.ndarray: A 3-channel RGB image (uint8) with shape (3, H, W), where
                    each channel is a copy of the input grayscale image.
    """
    if not isinstance(greyscale_img, np.ndarray):
        greyscale_img = np.array(greyscale_img)

    if len(greyscale_img.shape) > 2:
        if greyscale_img.shape[0] !=3 and greyscale_img.shape[-1] == 3: # HxWxC
            return np.transpose(greyscale_img, (2,0,1))
        return greyscale_img

    if greyscale_img.dtype != np.uint8: # Ensure uint8 for stacking
        if greyscale_img.max() <=1.0 and greyscale_img.min() >=0: # Probably float 0-1 range
            greyscale_img = (greyscale_img * 255).astype(np.uint8)
        else: # Assume it's some other int type, clip and convert
            greyscale_img = np.clip(greyscale_img, 0, 255).astype(np.uint8)

    # Expand dims to (1, H, W) then concatenate along axis 0
    return np.concatenate([np.expand_dims(greyscale_img, axis=0)] * 3, axis=0)
    

if __name__ == "__main__":
    import cv2
    ramp=np.arange(0,TWO_TO_TWENTYFORUTH_MINUS_ONE,50000)
    n=ramp.shape[0]
    x,y=np.meshgrid(ramp,ramp)
    x_rb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    y_rb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    x_rgb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    y_rgb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            print('[INFO] i:',str(i),', j:',str(j))
            x_rb[i,j]=convert_to_rainbow(x[i,j])
            y_rb[i,j]=convert_to_rainbow(y[i,j])
            x_rgb[i,j]=convert_to_rgb(x[i,j])
            y_rgb[i,j]=convert_to_rgb(y[i,j])
    
    cv2.imshow('x rainbow',cv2.cvtColor(x_rb,cv2.COLOR_RGB2BGR))
    cv2.imshow('y rainbow',cv2.cvtColor(y_rb,cv2.COLOR_RGB2BGR))
    cv2.imshow('x rgb',cv2.cvtColor(x_rgb,cv2.COLOR_RGB2BGR))
    cv2.imshow('y rgb',cv2.cvtColor(y_rgb,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

