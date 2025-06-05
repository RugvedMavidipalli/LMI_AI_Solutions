"""
Provides utility functions for image alignment, contour processing, and region extraction.

This module includes functions for:
- Detecting contours in images.
- Masking images using contours.
- Rotating images based on contour orientation (either by fitting an ellipse or using
  the minimum area bounding box).
- Cropping images based on contour bounding boxes.
- Extracting regions of interest (ROIs) defined in a JSON file.
- A combined alignment and cropping pipeline for images.
- Tiling an image into smaller, potentially overlapping, windows.

The primary image processing library used is OpenCV (cv2).
"""
# %% modules
import json
import numpy as np
import cv2
import math
import os

from image_utils.img_rotate import rotate

# %% convert intensity pcd to png


def intensitypcd_2_png() -> None:
    """
    Placeholder function for converting intensity PCD (Point Cloud Data) to PNG.

    Currently not implemented.
    """
    pass

# %% get image contours


def getContours(img: np.ndarray, blur: tuple = (17, 17), threshold: tuple = (20, 150)) -> list:
    """
    Finds and returns contours in an image.

    The image is first converted to grayscale, then blurred, and Canny edge
    detection is applied. Contours are found in the dilated Canny edges.

    Args:
        img (np.ndarray): The input image (expected in BGR format).
        blur (tuple, optional): Kernel size for Gaussian blur. Defaults to (17, 17).
        threshold (tuple, optional): Thresholds (T1, T2) for Canny edge detection.
                                     Defaults to (20, 150).

    Returns:
        list: A list of contours found in the image, sorted by area.
              Each contour is a NumPy array of (x,y) coordinates.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, blur, 0)
    T1 = threshold[0]
    T2 = threshold[1]
    canny = cv2.Canny(blurred, T1, T2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    dilated = cv2.dilate(canny, kernel)
    cv2.imshow('dilated', dilated)
    cv2.waitKey(2000)
    (contours, _) = cv2.findContours(dilated.copy(),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    return contours

# %% mask image by contour


def maskByContour(image: np.ndarray, contour: list) -> np.ndarray:
    """
    Masks an image using a given contour.

    Creates a binary mask from the contour and applies it to the image.
    Only the area within the contour will remain in the output image.

    Args:
        image (np.ndarray): The input image.
        contour (list): A list of contours (or a single contour) to use for masking.
                        If a list, all contours are drawn on the mask.

    Returns:
        np.ndarray: The masked image.
    """
    mask = np.zeros(image.shape[:2], dtype='uint8')
    # cv2.drawContours expects a list of contours. If a single contour (np.array) is passed, wrap it.
    contours_list = [contour] if isinstance(contour, np.ndarray) else contour
    cv2.drawContours(mask, contours_list, -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

# %% rotate by contour


def rotateByContour(image: np.ndarray, contour: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Rotates an image based on the orientation of an ellipse fitted to a contour.

    The rotation angle is determined by `cv2.fitEllipse`. The image is rotated
    around the center of the contour.

    Args:
        image (np.ndarray): The input image.
        contour (np.ndarray): The contour whose orientation will guide the rotation.

    Returns:
        tuple[int, int, np.ndarray]:
            - cx (int): The x-coordinate of the contour center.
            - cy (int): The y-coordinate of the contour center.
            - rotated (np.ndarray): The rotated image.
    """
    if contour is None or len(contour) < 5: # cv2.fitEllipse requires at least 5 points
        # Return original image and its center if contour is not suitable
        h, w = image.shape[:2]
        return w // 2, h // 2, image

    _, _, angle = cv2.fitEllipse(contour)
    M = cv2.moments(contour)
    if M['m00'] == 0: # Avoid division by zero if contour area is zero
        h, w = image.shape[:2]
        return w // 2, h // 2, image
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print('minBB angle ={}'.format(angle))
    #print('centerX = {}'.format(cx) + ' centerY = {}'.format(cy))
    rotated = rotate(image, -(180-angle), center=(cx, cy)) # Assumes 'rotate' is an available function
    return cx, cy, rotated

# %% get contour center


def getContCenter(contour: np.ndarray) -> tuple[int, int]:
    """
    Calculates the center (centroid) of a contour using image moments.

    Args:
        contour (np.ndarray): The input contour.

    Returns:
        tuple[int, int]: (cx, cy), the x and y coordinates of the contour's centroid.
                         Returns (0,0) if the contour moment m00 is zero to avoid division by zero.
    """
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return 0, 0
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy

# %% rotate image by minimum contour bounding box


def rotateByMinBB(image: np.ndarray, contour: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Rotates an image based on the angle of the minimum area bounding box of a contour.

    The goal is to align the contour's longer dimension with the vertical axis.
    The image is rotated around the center of the contour.

    Args:
        image (np.ndarray): The input image.
        contour (np.ndarray): The contour to use for calculating the rotation angle.

    Returns:
        tuple[int, int, np.ndarray]:
            - cx (int): The x-coordinate of the contour center.
            - cy (int): The y-coordinate of the contour center.
            - rotated (np.ndarray): The rotated image.
    """
    M = cv2.moments(contour)
    if M['m00'] == 0: # Avoid division by zero
        h, w = image.shape[:2]
        return w // 2, h // 2, image
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    delx30 = box[3][0]-box[0][0]
    dely30 = box[3][1]-box[0][1]
    delx10 = box[1][0]-box[0][0]
    dely10 = box[1][1]-box[0][1]
    angle = np.arctan(dely30/delx30)*180/np.pi
    da = 180
    # Check rotation for long edge or short edge, want long edge up
    mag30 = np.sqrt(delx30*delx30+dely30*dely30)
    mag10 = np.sqrt(delx10*delx10+dely10*dely10)
    if mag30 < mag10:
        da += 90
    rotated = rotate(image, angle+da, center=(cx, cy))
    return cx, cy, rotated

# %% crop image by contour bounding box


def cropByBB(image: np.ndarray, contour: np.ndarray, delta: int = 0) -> np.ndarray:
    """
    Crops an image based on the bounding box of a contour.

    The bounding box is determined by `cv2.minAreaRect`. The cropping region
    can be adjusted by the `delta` parameter (padding or shrinking).

    Args:
        image (np.ndarray): The input image.
        contour (np.ndarray): The contour whose bounding box defines the crop area.
        delta (int, optional): An offset to adjust the crop boundaries.
                               Positive values shrink the box, negative values expand it.
                               Defaults to 0.

    Returns:
        np.ndarray: The cropped image. Returns an empty array if coordinates are invalid.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    (img_h, img_w) = image.shape[:2]

    # Get min/max x and y from the bounding box points
    # Add delta for padding: positive delta shrinks, negative delta expands
    # Ensure coordinates are within image boundaries
    xl = max(0, np.min(box[:, 0]) + delta)
    xr = min(img_w, np.max(box[:, 0]) - delta)
    yt = max(0, np.min(box[:, 1]) + delta)
    yb = min(img_h, np.max(box[:, 1]) - delta)

    if xr <= xl or yb <= yt: # Check for invalid or zero-size crop
        return np.array([]) # Return empty image

    newImage = image[yt:yb, xl:xr]
    return newImage


# %% extract box regions from JSON file
def extract_UniformBox_ROI_from_JSON(json_file_path: str,
                                     input_image_dir_path: str,
                                     output_image_dir_path: str,
                                     render_defects: bool = False) -> int:
    """
    Extracts regions of interest (ROIs) from images based on a JSON file.

    The JSON file should define regions with 'x', 'y', 'width', 'height' attributes.
    This function calculates a uniform square window size based on the maximum
    width and height of all defined ROIs. It then crops these square ROIs from
    the corresponding images and saves them.

    Args:
        json_file_path (str): Path to the JSON file defining ROIs.
        input_image_dir_path (str): Path to the directory containing the source images.
        output_image_dir_path (str): Path to the directory where cropped ROI images
                                     will be saved.
        render_defects (bool, optional): If True, displays images with bounding boxes
                                         drawn around the extracted ROIs. Defaults to False.

    Returns:
        int: The dimension (wd) of the uniform square window used for cropping.
    """
    # computes the max of a list of lists
    def max_list_of_lists(myList: list) -> float: # Assuming numbers, could be int or float
        if not myList: return 0
        n = len(myList)
        myMax = [None]*n
        for i in range(n):
            myMax[i] = max(myList[i])
        return max(myMax)

    # parse the labeling data
    with open(json_file_path, 'r') as read_file:
        defect_labels = json.load(read_file)

    # keys for each region
    keys = list(defect_labels.keys())

    # extract top left corner and box dimensions
    xx = [None]*len(keys)
    yy = [None]*len(keys)
    ww = [None]*len(keys)
    hh = [None]*len(keys)
    for i, key in enumerate(keys):
        regions = defect_labels[key]['regions']
        xj = [None]*len(regions)
        yj = [None]*len(regions)
        wj = [None]*len(regions)
        hj = [None]*len(regions)
        for j, region in enumerate(regions):
            xj[j] = regions[j]['shape_attributes']['x']
            yj[j] = regions[j]['shape_attributes']['y']
            wj[j] = regions[j]['shape_attributes']['width']
            hj[j] = regions[j]['shape_attributes']['height']
        xx[i] = xj
        yy[i] = yj
        ww[i] = wj
        hh[i] = hj
    # get maximum box dimension
    w_max = max_list_of_lists(ww)
    h_max = max_list_of_lists(hh)
    print('[INFO] maximum width ', w_max)
    print('[INFO] maximum height ', h_max)
    # compute square window dimension
    wd = max((w_max, h_max))
    print('[INFO] window dimension ', wd)

    # write defect images
    if not os.path.isdir(output_image_dir_path):
        os.mkdir(output_image_dir_path)
    for i, key in enumerate(keys):
        fname = input_image_dir_path+'/'+defect_labels[key]['filename']
        image = cv2.imread(fname)
        xi = xx[i]
        yi = yy[i]
        wi = ww[i]
        hi = hh[i]
        for j in range(len(xi)):
            xj = xi[j]-(wd-wi[j])/2
            xj = math.floor(xj)
            yj = yi[j]-(wd-hi[j])/2
            yj = math.floor(yj)
            cropped = image[yj:yj+wd, xj:xj+wd]
            fcrop = output_image_dir_path+'/' + \
                os.path.splitext(os.path.split(fname)[1])[0]+'_'+str(j)+'.png'
            cv2.imwrite(fcrop, cropped)
    if render_defects:
        for i, key in enumerate(keys):
            fname = os.path.split(json_file_path)[
                0]+'/'+defect_labels[key]['filename']
            image = cv2.imread(fname)
            green = (0, 255, 0)
            xi = xx[i]
            yi = yy[i]
            wi = ww[i]
            hi = hh[i]
            for j in range(len(xi)):
                xj = xi[j]-(wd-wi[j])/2
                xj = math.floor(xj)
                yj = yi[j]-(wd-hi[j])/2
                yj = math.floor(yj)
                cv2.rectangle(image, (xj, yj), (xj+wd, yj+wd), green, 3)
            cv2.imshow('image', image)
            cv2.waitKey(5000)

    return wd


def align_and_crop(image_file_path: str, output_dir_path: str) -> None:
    """
    Performs a sequence of alignment and cropping operations on an image.

    The process involves:
    1. Adding a border to the image to ensure outer contours can be found.
    2. Getting the primary contour of the bordered image.
    3. Rotating the image based on the minimum area bounding box of this contour.
    4. Getting the contour of the rotated image.
    5. Cropping the rotated image based on the new contour's bounding box (with a small shrink delta).
    6. Saving the final cropped image.

    Args:
        image_file_path (str): Path to the input image file.
        output_dir_path (str): Path to the directory where the cropped image will be saved.
                               The output filename will be <original_filename>_cropped.png.
    """
    # import image
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_file_path}")
        return

    # add border around image that will enable outer contour extraction
    # expand top and bottom for vertical alignment
    shape_init = image.shape
    borderLR = 50
    borderTB = borderLR
    if shape_init[1] > shape_init[0]:
        borderTB = shape_init[1]-shape_init[0]+borderTB
    image = cv2.copyMakeBorder(
        image, borderTB, borderTB, borderLR, borderLR, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # extract outer contour
    contour = getContours(image)
    print('[INFO]: {0:2d} contours'.format(len(contour)))
    # rotate the image
    _, _, rotated = rotateByMinBB(image, contour[0])
    # get new contour
    contour = getContours(rotated)
    print('[INFO] {0:2d} contours'.format(len(contour)))
    # crop by bounding box
    cropped = cropByBB(rotated, contour[0], 15)
    # save the cropped image
    if not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    fcrop = output_dir_path+'/' + \
        os.path.splitext(os.path.split(image_file_path)[1])[0]+'_cropped.png'
    cv2.imwrite(fcrop, cropped)
    print(f"[INFO] Saved aligned and cropped image to: {fcrop}")


def tile_image(image_file_path: str, tile_dir_path: str, window_dimension: int, steps_per_window: int = 2) -> None:
    """
    Tiles an image into smaller, potentially overlapping, windows and saves them.

    Args:
        image_file_path (str): Path to the input image file.
        tile_dir_path (str): Path to the directory where the tiled images will be saved.
        window_dimension (int): The width and height (square window) of each tile.
        steps_per_window (int, optional): Determines the overlap between tiles.
                                          A value of 2 means a 50% overlap (step size is wd/2).
                                          A value of 1 means no overlap (step size is wd).
                                          Defaults to 2.
    """
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_file_path}")
        return

    wd = window_dimension
    if wd <= 0:
        print("[ERROR] Window dimension must be positive.")
        return
    if steps_per_window <= 0:
        print("[ERROR] Steps per window must be positive.")
        return

    step_size = wd / steps_per_window
    if step_size == 0: # Avoid infinite loop if wd is small and steps_per_window is large
        print("[ERROR] Calculated step size is zero. Adjust window_dimension or steps_per_window.")
        return

    num_steps_y = int(image.shape[0] // step_size)
    num_steps_x = int(image.shape[1] // step_size)

    if not os.path.isdir(tile_dir_path):
        print(f"[INFO] Creating tile directory: {tile_dir_path}")
        os.makedirs(tile_dir_path)

    base_filename = os.path.splitext(os.path.split(image_file_path)[1])[0]

    for j in range(num_steps_y - (steps_per_window -1) if steps_per_window > 0 else num_steps_y ): # Adjust loop range to avoid going out of bounds for last tile
        for i in range(num_steps_x - (steps_per_window -1) if steps_per_window > 0 else num_steps_x ):
            y_start = int(j * step_size)
            x_start = int(i * step_size)

            y_end = y_start + wd
            x_end = x_start + wd

            # Ensure tile does not exceed image boundaries if we are at the edge
            if y_end > image.shape[0]: y_end = image.shape[0]; y_start = y_end - wd
            if x_end > image.shape[1]: x_end = image.shape[1]; x_start = x_end - wd
            if y_start < 0: y_start = 0 # Should not happen with current loop logic but good for safety
            if x_start < 0: x_start = 0


            windowed_tile = image[y_start:y_end, x_start:x_end]

            if windowed_tile.size == 0: # Skip if tile is empty (e.g. due to rounding or extreme params)
                print(f"[WARNING] Empty tile generated for i={i}, j={j}. Skipping.")
                continue

            # print(f'[INFO] Tile (i={i},j={j}): x_range=({x_start}:{x_end}), y_range=({y_start}:{y_end}), shape={windowed_tile.shape}')

            tile_filename = os.path.join(tile_dir_path, f'{base_filename}_tile_{i}_{j}.png')
            try:
                cv2.imwrite(tile_filename, windowed_tile)
            except Exception as e:
                print(f"[ERROR] Could not save tile {tile_filename}: {e}")
