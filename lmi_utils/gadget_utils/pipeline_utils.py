"""
A collection of utility functions for image and data processing pipelines,
particularly focused on tasks involving image manipulation, coordinate transformations,
and data loading for machine learning models.

This module includes functions for:
- Image resizing and padding using PyTorch or OpenCV.
- Conversion between uint16 and int16 profile data.
- Transforming 2D profile data to 3D point clouds.
- Converting 2D pixel coordinates to 3D sensor space.
- Plotting bounding boxes (standard and rotated) and masks on images.
- Reverting and applying sequences of geometric operations on points and masks.
- Helper functions for dictionary manipulation (key conversion, value-to-key).
- Batch loading of image paths or gadget image (profile/intensity) pairs.
- Loading pipeline configuration definitions from JSON files.

Dependencies include NumPy, OpenCV (cv2), PyTorch, and standard Python libraries
like os, json, random, logging, and glob.
"""
import numpy as np
import cv2
import random
import os
import json
import torch
import logging
import glob
from torch.nn import functional as F


BLACK=(0,0,0)
TWO_TO_FIFTEEN = 2**15

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.no_grad()
def resize_image(im: np.ndarray | torch.Tensor, W: int = None, H: int = None, mode: str = 'bilinear') -> np.ndarray | torch.Tensor:
    """
    Resizes an image to a target width (W) and height (H) using PyTorch's F.interpolate.

    If only W or H is provided, the other dimension is scaled to maintain aspect ratio.
    If neither W nor H is provided, the original image is returned.

    Args:
        im (np.ndarray | torch.Tensor): The input image of shape (H,W) or (H,W,C).
        W (int, optional): Target width. Defaults to None.
        H (int, optional): Target height. Defaults to None.
        mode (str, optional): Interpolation mode. Options include 'nearest', 'linear',
                              'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact'.
                              Defaults to 'bilinear'.

    Returns:
        np.ndarray | torch.Tensor: The resized image, in the same format (NumPy array or
                                   PyTorch tensor) as the input.
    """
    if W is None and H is None:
        return im
    
    # get the target width and height
    h,w = im.shape[:2]
    if W is None:
        W = int(w*H/h)
    elif H is None:
        H = int(h*W/w)
    
    # convert to tensor
    is_numpy = isinstance(im, np.ndarray)
    if is_numpy:
        im = torch.from_numpy(im)
    
    # deal with 1 channel image 
    one_channel = im.ndim==2
    if one_channel:
        im = im.unsqueeze(-1)
        
    im2 = F.interpolate(im.permute(2,0,1).unsqueeze(0).float(), size=(H,W), mode=mode)
    im2 = im2.squeeze(0).permute(1,2,0).to(torch.uint8)
    
    # back to 1 channel
    if one_channel:
        im2 = im2.squeeze(-1)
    
    return im2.numpy() if is_numpy else im2


@torch.no_grad()
def fit_im_to_size(im: np.ndarray | torch.Tensor, W: int = None, H: int = None, value: int = 0) -> tuple:
    """
    Pads or crops an image to a target size [W, H] using PyTorch's F.pad.

    Padding is applied symmetrically (half on each side). If cropping, it crops from
    the center. If W or H is None, the respective dimension is not changed.

    Args:
        im (np.ndarray | torch.Tensor): The input image of shape (H,W) or (H,W,C).
        W (int, optional): Target width. If None, original width is used. Defaults to None.
        H (int, optional): Target height. If None, original height is used. Defaults to None.
        value (int, optional): The value used for padding. Defaults to 0 (black).

    Returns:
        tuple:
            - im_processed (np.ndarray | torch.Tensor): The padded/cropped image, in the
              same format as input.
            - pad_L (int): Number of pixels padded (positive) or cropped (negative) from the left.
            - pad_R (int): Number of pixels padded (positive) or cropped (negative) from the right.
            - pad_T (int): Number of pixels padded (positive) or cropped (negative) from the top.
            - pad_B (int): Number of pixels padded (positive) or cropped (negative) from the bottom.
    """
    
    if W is None and H is None:
        return im, 0, 0, 0, 0
    h,w = im.shape[:2]
    if W is None:
        W = w
    elif H is None:
        H = h
    
    is_numpy = isinstance(im, np.ndarray)
    if is_numpy:
        im = torch.from_numpy(im)

    # deal with 1 channel image
    one_channel = im.ndim==2
    if one_channel:
        im = im.unsqueeze(-1)

    # convert to CHW format    
    im = im.permute(2, 0, 1)

    # pad/crop width
    if W >= w:
        pad_L = (W - w) // 2
        pad_R = W - w - pad_L
        im = F.pad(im, (pad_L, pad_R, 0, 0), value=value)  
    else:
        pad_L = (w - W) // 2
        pad_R = w - W - pad_L
        im = im[:, :, pad_L:-pad_R]
        pad_L *= -1
        pad_R *= -1

    # pad/crop height
    if H >= h:
        pad_T = (H - h) // 2
        pad_B = H - h - pad_T
        im = F.pad(im, (0, 0, pad_T, pad_B), value=value)
    else:
        pad_T = (h - H) // 2
        pad_B = h - H - pad_T
        im = im[:, pad_T:-pad_B, :]
        pad_T *= -1
        pad_B *= -1

    # convert back to HWC format
    im = im.permute(1, 2, 0)
    
    # back to 1 channel
    if one_channel:
        im = im.squeeze(-1)

    if is_numpy:
        im = im.numpy()
    return im, pad_L, pad_R, pad_T, pad_B


def fit_array_to_size(im: np.ndarray, W: int = None, H: int = None, value: int = 0) -> tuple:
    """
    Pads or crops a NumPy image array to a target size [W, H] using OpenCV's copyMakeBorder.

    This function is an alternative to `fit_im_to_size` for NumPy arrays, using OpenCV
    for padding. Behavior for padding/cropping is similar: symmetrical.

    Args:
        im (np.ndarray): The input image array of shape (H,W) or (H,W,C).
        W (int, optional): Target width. If None, original width is used. Defaults to None.
        H (int, optional): Target height. If None, original height is used. Defaults to None.
        value (int, optional): The value used for padding. Defaults to 0 (black).

    Returns:
        tuple:
            - im_processed (np.ndarray): The padded/cropped image.
            - pad_L (int): Number of pixels padded/cropped from the left.
            - pad_R (int): Number of pixels padded/cropped from the right.
            - pad_T (int): Number of pixels padded/cropped from the top.
            - pad_B (int): Number of pixels padded/cropped from the bottom.
    """
    h_im,w_im=im.shape[:2]
    if W is None:
        W=w_im
    if H is None:
        H=h_im
    # pad or crop width
    if W >= w_im:
        pad_L=(W-w_im)//2
        pad_R=W-w_im-pad_L
        im=cv2.copyMakeBorder(im,0,0,pad_L,pad_R,cv2.BORDER_CONSTANT,value)
    else:
        pad_L = (w_im-W)//2
        pad_R = w_im-W-pad_L
        im = im[:,pad_L:-pad_R]
        pad_L *= -1
        pad_R *= -1
    # pad or crop height
    if H >= h_im:
        pad_T=(H-h_im)//2
        pad_B=H-h_im-pad_T
        im=cv2.copyMakeBorder(im,pad_T,pad_B,0,0,cv2.BORDER_CONSTANT,value)
    else:
        pad_T = (h_im-H)//2
        pad_B = h_im-H-pad_T
        im = im[pad_T:-pad_B,:]
        pad_T *= -1
        pad_B *= -1
    return im, pad_L, pad_R, pad_T, pad_B


def uint16_to_int16(profile: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts a uint16 profile image to int16 format.

    This is often needed as profile data might be stored as uint16 (e.g., by adding 2^15)
    but needs to be in int16 for certain calculations or interpretations where
    -2^15 represents invalid/missing data.

    Args:
        profile (np.ndarray | torch.Tensor): The input uint16 profile image.

    Returns:
        np.ndarray | torch.Tensor: The profile image converted to int16, in the
                                   same format (NumPy or PyTorch) as input.

    Raises:
        Exception: If the input profile is not uint16.
    """
    is_numpy = isinstance(profile, np.ndarray)
    if is_numpy:
        profile = torch.from_numpy(profile)
        
    if profile.dtype != torch.uint16:
        raise Exception(f'input should be uint16, got {profile.dtype}')
    
    profile = profile.to(torch.int32) - torch.tensor(TWO_TO_FIFTEEN,dtype=torch.int32)
    profile = profile.to(torch.int16)
    return profile.numpy() if is_numpy else profile


@torch.no_grad()
def profile_to_3d(profile: np.ndarray | torch.Tensor, resolution: tuple, offset: tuple) -> tuple:
    """
    Converts a 2D profile image (height map) to 3D sensor space coordinates (X, Y, Z).

    Also returns a mask indicating valid (non-background) points in the profile.
    Assumes that -2^15 in the int16 profile represents invalid/background points.

    Args:
        profile (np.ndarray | torch.Tensor): The input int16 profile image.
        resolution (tuple): A tuple (x_resolution, y_resolution, z_resolution)
                            defining the scaling factor for each dimension.
        offset (tuple): A tuple (x_offset, y_offset, z_offset) defining the
                        origin offset for each dimension.

    Returns:
        tuple:
            - X (np.ndarray | torch.Tensor): 2D array of X coordinates in 3D space,
              same shape as input profile.
            - Y (np.ndarray | torch.Tensor): 2D array of Y coordinates in 3D space.
            - Z (np.ndarray | torch.Tensor): 2D array of Z coordinates (heights) in 3D space.
            - mask (np.ndarray | torch.Tensor): A boolean 2D array indicating valid points
              (True where profile data is not -2^15).
              All returned tensors/arrays match the input type (NumPy or PyTorch).

    Raises:
        Exception: If the input profile is not int16.
    """
    is_numpy = isinstance(profile, np.ndarray)
    
    # convert to tensor
    if is_numpy:
        profile = torch.from_numpy(profile)
    resolution = torch.from_numpy(np.array(resolution)).to(profile.device)
    offset = torch.from_numpy(np.array(offset)).to(profile.device)
        
    if profile.dtype != torch.int16:
        raise Exception(f'profile.dtype should be int16, got {profile.dtype}')
    
    h,w = profile.shape[:2]
    x1,y1 = 0,0
    x2,y2 = w,h
    mask = profile != -TWO_TO_FIFTEEN
    x_range = torch.arange(x1,x2,device=profile.device)
    y_range = torch.arange(y1,y2,device=profile.device)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='xy')
    X = offset[0] + xx * resolution[0]
    Y = offset[1] + yy * resolution[1]
    Z = offset[2] + profile*resolution[2]
    
    if is_numpy:
        X = X.numpy()
        Y = Y.numpy()
        Z = Z.numpy()
        mask = mask.numpy()
    return X,Y,Z,mask


def pts_to_3d(pts: np.ndarray | torch.Tensor, profile: np.ndarray | torch.Tensor,
              resolution: tuple, offset: tuple) -> np.ndarray | torch.Tensor:
    """
    Converts a list of 2D pixel locations (points) to 3D sensor space coordinates.

    Args:
        pts (np.ndarray | torch.Tensor): An Nx2 array of (x, y) pixel coordinates.
        profile (np.ndarray | torch.Tensor): The int16 profile image corresponding to the points.
                                            Used to get Z values.
        resolution (tuple): (x_resolution, y_resolution, z_resolution) for scaling.
        offset (tuple): (x_offset, y_offset, z_offset) for origin offset.

    Returns:
        np.ndarray | torch.Tensor: An Nx3 array of (X, Y, Z) coordinates in 3D sensor space,
                                   matching the input type (NumPy or PyTorch).

    Raises:
        Exception: If input types of `pts` and `profile` do not match, or if `pts`
                   has an incorrect shape, or if `profile` is not int16.
    """
    if type(pts) != type(profile):
        raise Exception(f'pts and profile should have the same type, got {type(pts)} and {type(profile)}')
    
    is_numpy = isinstance(pts, np.ndarray)
    if is_numpy:
        pts = torch.from_numpy(pts)
        profile = torch.from_numpy(profile)
    offset = torch.from_numpy(np.array(offset)).to(pts.device)
    resolution = torch.from_numpy(np.array(resolution)).to(pts.device)
    
    if pts.device != profile.device:
        raise Exception(f'device of pts and profile should be the same, got {pts.device} and {profile.device}')
    if pts.ndim!=2 or pts.shape[1]!=2:
        raise Exception(f'shape of pts should be Nx2, got {pts.shape}')
    if profile.dtype != torch.int16:
        raise Exception(f'profile.dtype should be int16, got {profile.dtype}')
    
    pts = pts.to(torch.int32)
    Xs = pts[:,0]
    Ys = pts[:,1]
    
    nx = offset[0] + Xs * resolution[0]
    ny = offset[1] + Ys * resolution[1]
    nz = offset[2] + profile[Ys,Xs]*resolution[2]
    xyz = torch.stack([nx,ny,nz],dim=1)
    
    return xyz.numpy() if is_numpy else xyz


def plot_one_box(box: np.ndarray | list,
                 img: np.ndarray,
                 mask: np.ndarray | torch.Tensor = None,
                 mask_threshold: float = 0.0,
                 color: list = None,
                 label: str = None,
                 line_thickness: int = None,
                 hide_bbox: bool = False):
    """
    Plots one bounding box (and optionally a mask) on an image.

    This function is adapted from common plotting utilities in object detection projects.
    The image is modified in-place.

    Args:
        box (np.ndarray | list): Bounding box coordinates, expected as [x1, y1, x2, y2].
        img (np.ndarray): The image (OpenCV BGR format) on which to draw.
        mask (np.ndarray | torch.Tensor, optional): A binary or probability mask
                                                    corresponding to the bounding box.
                                                    If provided, it's drawn on the image.
                                                    Defaults to None.
        mask_threshold (float, optional): Threshold for binarizing the mask if it's not
                                          already binary. Defaults to 0.0.
        color (list, optional): Color for the bounding box and mask, as a BGR tuple
                                (e.g., [0, 255, 0] for green). If None, a random
                                color is used. Defaults to None.
        label (str, optional): A label string to display with the bounding box.
                               Defaults to None.
        line_thickness (int, optional): Thickness of the bounding box lines. If None,
                                        it's auto-calculated based on image size.
                                        Defaults to None.
        hide_bbox (bool, optional): If True, does not draw the bounding box rectangle
                                    (e.g., if only drawing a mask). Defaults to False.
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    if isinstance(box, list):
        box = np.array([x for x in box])
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
        
    x1,y1,x2,y2 = box.astype(int)
    c1, c2 = (x1, y1), (x2, y2)
    if not hide_bbox:
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if mask is not None:
        # mask *= 255
        m = mask>mask_threshold
        blended = (0.4 * np.array(color,dtype=float) + 0.6 * img[m]).astype(np.uint8)
        img[m] = blended
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_one_rbox(box: np.ndarray | list,
                  img: np.ndarray,
                  color: list = None,
                  label: str = None,
                  line_thickness: int = None,
                  hide_bbox: bool = False):
    """
    Plots one rotated bounding box on an image.

    The image is modified in-place.

    Args:
        box (np.ndarray | list): Rotated bounding box coordinates, expected as a list
                                 or array of 4 points, e.g., [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        img (np.ndarray): The image (OpenCV BGR format) on which to draw.
        color (list, optional): Color for the bounding box, as a BGR tuple.
                                If None, a random color is used. Defaults to None.
        label (str, optional): A label string to display with the bounding box.
                               Defaults to None.
        line_thickness (int, optional): Thickness of the bounding box lines. If None,
                                        it's auto-calculated. Defaults to None.
        hide_bbox (bool, optional): If True, does not draw the bounding box polygon.
                                    Defaults to False.

    Raises:
        Exception: If the input `box` does not contain 4 points.
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    if len(box) != 4:
        raise Exception(f'box should be a list of 4 points, got {len(box)} points')
    if isinstance(box, list):
        box = np.array(box)
    box = box.astype(int)
    
    if not hide_bbox:
        cv2.polylines(img, [box], isClosed=True, color=color, thickness=tl)
        
    if label:
        highest_point = min(box, key=lambda point: point[1])
        text_position = (highest_point[0], highest_point[1] - 10)
        
        if text_position[1] < 0:  # If the text would be outside the image, move it below the lowest point instead
            lowest_point = max(box, key=lambda point: point[1])
            text_position = (lowest_point[0], lowest_point[1] + 20)
        
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        cv2.rectangle(img, text_position, (text_position[0] + t_size[0], text_position[1] - t_size[1] - 3), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            text_position,
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


@torch.no_grad()
def revert_mask_to_origin(mask: np.ndarray | torch.Tensor, operations: list) -> np.ndarray | torch.Tensor:
    """
    Reverts transformations applied to a single mask image to restore its original state.

    The transformations are reverted in the reverse order of their application,
    as defined in the `operations` list. Supported operations: 'resize', 'pad', 'flip'.

    Args:
        mask (np.ndarray | torch.Tensor): The mask image to transform, shape (H,W) or (H,W,C).
        operations (list): A list of dictionaries, where each dictionary defines a
                           transformation and its parameters.
                           - {'resize': [target_w, target_h, original_w, original_h]}
                           - {'pad': [pad_left, pad_right, pad_top, pad_bottom]} (values are amounts padded)
                           - {'flip': [lr_flipped (bool), ud_flipped (bool), im_width, im_height]}

    Returns:
        np.ndarray | torch.Tensor: The mask image with transformations reverted,
                                   in the same format as input.
    """
    is_numpy = isinstance(mask, np.ndarray)
    for operator in reversed(operations):
        if 'resize' in operator:
            _,_,nw,nh = operator['resize']
            mask = resize_image(mask,nw,nh)
        if 'pad' in operator:
            h,w = mask.shape[:2]
            pad_L,pad_R,pad_T,pad_B = operator['pad']
            nw,nh = w-pad_L-pad_R,h-pad_T-pad_B
            mask,_,_,_,_ = fit_im_to_size(mask,nw,nh)
        if 'flip' in operator:
            lr,ud,im_w,im_h = operator['flip']
            if is_numpy:
                mask = torch.from_numpy(mask)
            if lr:
                mask = torch.flip(mask,[1])
            if ud:
                mask = torch.flip(mask,[0])
            if is_numpy:
                mask = mask.numpy()
    return mask


def revert_masks_to_origin(masks: list | np.ndarray | torch.Tensor, operations: list) -> list | np.ndarray | torch.Tensor:
    """
    Reverts transformations for a list or batch of mask images.

    Args:
        masks (list | np.ndarray | torch.Tensor): A list or batch of mask images.
        operations (list): A list of transformation operation dictionaries, as defined
                           in `revert_mask_to_origin`.

    Returns:
        list | np.ndarray | torch.Tensor: The transformed masks, in the same batch
                                          format as input (list, NumPy array, or PyTorch tensor).
    """
    results = []
    if len(masks)==0:
        return masks # Return empty input as is

    # Determine input type to return same type
    is_torch_tensor_input = isinstance(masks, torch.Tensor)
    # Handle cases where masks might be a list of tensors vs a single stacked tensor
    # For now, assumes if torch.Tensor, it's a batch that can be iterated.
    # If it's a list of tensors/arrays, it will also work.

    is_numpy_array_input = isinstance(masks, np.ndarray)
    # Check if the first element is a tensor if input is a list
    is_list_of_tensors = isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], torch.Tensor)


    for m in masks:
        results.append(revert_mask_to_origin(m, operations))

    if is_torch_tensor_input or is_list_of_tensors:
        return torch.stack(results) if results else torch.empty(0)
    elif is_numpy_array_input:
        return np.stack(results) if results else np.empty((0,) + masks.shape[1:]) # preserve ndim if empty
    return results # Return as list if input was list of non-tensors


@torch.no_grad()
def revert_to_origin(pts: np.ndarray | torch.Tensor | list, operations: list) -> np.ndarray | torch.Tensor | list:
    """
    Reverts a sequence of geometric transformations applied to a set of points.

    The transformations are reverted in the reverse order of their application as
    defined in the `operations` list. Supported operations: 'resize', 'pad',
    'stretch', 'flip'.

    Args:
        pts (np.ndarray | torch.Tensor | list): An Nx2 or Nx4 array/tensor/list of points.
                                                Each row is (x,y) or (x1,y1,x2,y2).
        operations (list): A list of dictionaries defining transformations.
                           - {'resize': [target_w, target_h, original_w, original_h]}
                           - {'pad': [pad_left, pad_right, pad_top, pad_bottom]}
                           - {'stretch': [stretch_x_ratio, stretch_y_ratio]}
                           - {'flip': [lr_flipped (bool), ud_flipped (bool), im_width, im_height]}

    Returns:
        np.ndarray | torch.Tensor | list: The points with transformations reverted,
                                          rounded to nearest integer and clamped to be non-negative.
                                          Output type matches input type.

    Raises:
        Exception: If `pts` has an unsupported shape or an operation is unsupported.
    """
    is_tensor = isinstance(pts, torch.Tensor)
    is_numpy = isinstance(pts, np.ndarray)
    if not is_tensor:
        pts = torch.from_numpy(pts) if is_numpy else torch.as_tensor(pts)
    
    if pts.ndim!=2 or (pts.shape[1]!=2 and pts.shape[1]!=4):
        raise Exception(f'pts should be Nx2 or Nx4, got shape: {pts.shape}')
    
    r,c = pts.shape
    for op in reversed(operations):
        if 'resize' in op:
            tw,th,orig_w,orig_h = op['resize']
            r = torch.tensor([tw/orig_w,th/orig_h],device=pts.device)
            if c==4:
                r = r.repeat(2).unsqueeze(0)
            pts = pts/r
        elif 'pad' in op:
            pad_L,pad_R,pad_T,pad_B = op['pad']
            p = torch.tensor([pad_L,pad_T],device=pts.device)
            if c==4:
                p = p.repeat(2).unsqueeze(0)
            pts = pts - p
        elif 'stretch' in op:
            s = torch.tensor(op['stretch'],device=pts.device)
            if c==4:
                s = s.repeat(2).unsqueeze(0)
            pts = pts/s
        elif 'flip' in op:
            lr,ud,im_w,im_h = op['flip']
            idx = [0,2] if c==4 else [0]
            idy = [1,3] if c==4 else [1]
            if lr:
                pts[:,idx] = im_w - pts[:,idx]
            if ud:
                pts[:,idy] = im_h - pts[:,idy]
        else:
            raise Exception(f'unsupported operation: {op}')
            
    pts = pts.round().clamp(min=0)
    if is_tensor:
        return pts
    return pts.numpy() if is_numpy else pts.tolist()


def apply_operations(pts: np.ndarray | torch.Tensor | list, operations: list) -> np.ndarray | torch.Tensor | list:
    """
    Applies a sequence of geometric transformations to a set of points.

    This function effectively does the inverse of `revert_to_origin`.
    It constructs an inverse set of operations and then calls `revert_to_origin`.

    Args:
        pts (np.ndarray | torch.Tensor | list): An Nx2 or Nx4 array/tensor/list of points.
        operations (list): A list of dictionaries defining transformations to apply,
                           in the same format as for `revert_to_origin`.

    Returns:
        np.ndarray | torch.Tensor | list: The points with transformations applied.
                                          Output type matches input type.

    Raises:
        Exception: If an operation is unsupported.
    """
    new_ops = []
    for op in operations:
        if 'resize' in op:
            tw,th,orig_w,orig_h = op['resize']
            tmp = {'resize': [orig_w,orig_h,tw,th]}
        elif 'pad' in op:
            pad_L,pad_R,pad_T,pad_B = op['pad']
            tmp = {'pad': [-pad_L,-pad_R,-pad_T,-pad_B]}
        elif 'stretch' in op:
            sx,sy = op['stretch']
            tmp = {'stretch': [1/sx,1/sy]}
        elif 'flip' in op:
            lr,ud,im_w,im_h = op['flip']
            tmp = {'flip': [lr,ud,im_w,im_h]}
        else:
            raise Exception(f'unsupported operation: {op}')
        new_ops.append(tmp)
    return revert_to_origin(pts, new_ops[::-1])
    
    
    
def convert_key_to_int(dt: dict) -> dict:
    """
    Converts string keys in a dictionary to integers.

    Useful for class maps where class IDs might be read as strings but need
    to be integers for processing.

    Args:
        dt (dict): The input dictionary.

    Returns:
        dict: A new dictionary with keys converted to integers.
    """
    return {int(k):dt[k] for k in dt}  


def val_to_key(dt: dict) -> dict:
    """
    Swaps keys and values in a dictionary.

    If multiple keys have the same value, the behavior is undefined as one
    key-value pair will overwrite others during construction.

    Args:
        dt (dict): The input dictionary.

    Returns:
        dict: A new dictionary where original values are keys and original keys are values.
    """
    return {dt[k]:k for k in dt}

    
def get_img_path_batches(batch_size: int, img_dir: str, fmt: str = 'png') -> list:
    """
    Scans a directory for image files and groups their paths into batches.

    Args:
        batch_size (int): The number of image paths per batch.
        img_dir (str): The directory to scan for images.
        fmt (str, optional): The file extension (format) of images to find.
                             Defaults to 'png'.

    Returns:
        list: A list of lists, where each inner list contains a batch of image file paths.
    """
    ret = []
    batch = []
    cnt_images = 0
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if name.find(f'.{fmt}')==-1:
                continue
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
            cnt_images += 1
    logger.info(f'loaded {cnt_images} files')
    if len(batch) > 0:
        ret.append(batch)
    return ret


def get_gadget_img_batches(batch_size: int, profile_dir: str, intensity_dir: str, fmt: str = 'png') -> list:
    """
    Scans directories for profile and intensity image files, pairs them, and groups into batches.

    Assumes that profile and intensity images have corresponding names and can be matched
    after sorting their respective file lists.

    Args:
        batch_size (int): The number of profile/intensity pairs per batch.
        profile_dir (str): Directory containing profile images.
        intensity_dir (str): Directory containing intensity images.
        fmt (str, optional): File extension for both profile and intensity images.
                             Defaults to 'png'.

    Returns:
        list: A list of lists, where each inner list contains a batch of dictionaries.
              Each dictionary is of the form {'profile': path_to_profile, 'intensity': path_to_intensity}.
    """
    profile_list = glob.glob(os.path.join(profile_dir,"*."+fmt))
    intensity_list = glob.glob(os.path.join(intensity_dir,"*."+fmt))

    profile_list.sort()
    intensity_list.sort()

    if len(profile_list) != len(intensity_list):
        logger.warning(f"Mismatch in number of profile ({len(profile_list)}) and "
                       f"intensity ({len(intensity_list)}) images with format '{fmt}'. "
                       "Pairing will be based on the shorter list length.")

    ret = []
    batch = []
    cnt_images = 0
    for profile, intensity in zip(profile_list, intensity_list): # zip stops at the shorter list
        if len(batch) == batch_size:
            ret.append(batch)
            batch = []
        batch.append({"profile":profile, "intensity":intensity})
        cnt_images += 1
    logger.info(f'Loaded {cnt_images} paired profile/intensity files.')
    if len(batch) > 0:
        ret.append(batch)
    return ret


def load_pipeline_def(filepath: str) -> dict:
    """
    Loads pipeline configuration definitions from a JSON file.

    The JSON file is expected to have a top-level key 'configs_def', which
    is a list of dictionaries. Each dictionary in the list should have 'name'
    and 'default_value' keys.

    Args:
        filepath (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary where keys are 'name' from the JSON structure and
              values are their corresponding 'default_value'.
              Returns an empty dict if 'configs_def' is missing or file not found.

    Raises:
        FileNotFoundError: If the specified filepath does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        KeyError: If 'configs_def' key is missing or items lack 'name'/'default_value'.
    """
    try:
        with open(filepath) as f:
            dt_all = json.load(f)

        l = dt_all.get('configs_def', []) # Use .get for safer access
        kwargs = {}
        for dt in l:
            if 'name' in dt and 'default_value' in dt:
                kwargs[dt['name']] = dt['default_value']
            else:
                logger.warning(f"Skipping config item due to missing 'name' or 'default_value': {dt}")
        return kwargs
    except FileNotFoundError:
        logger.error(f"Pipeline definition file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing expected key in pipeline definition {filepath}: {e}")
        raise
