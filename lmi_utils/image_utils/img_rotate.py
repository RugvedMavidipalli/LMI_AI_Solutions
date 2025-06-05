"""
Provides a utility function to rotate an image.

This module contains a single function `rotate` which uses OpenCV to perform
the rotation transformation.
"""
import numpy as np
import cv2

def rotate(image: np.ndarray, angle: float, center: tuple[int, int] = None, scale: float = 1.0) -> np.ndarray:
    """
    Rotates an image by a specified angle around a given center.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        angle (float): The angle of rotation in degrees. Positive values mean
                       counter-clockwise rotation (the coordinate origin is assumed
                       to be top-left).
        center (tuple[int, int], optional): The center of rotation (x, y).
                                            If None, the center of the image is used.
                                            Defaults to None.
        scale (float, optional): An isotropic scale factor for the image during rotation.
                                 Defaults to 1.0 (no scaling).

    Returns:
        np.ndarray: The rotated image. The output image will have the same
                    dimensions as the input image; parts of the rotated image
                    that fall outside these bounds will be clipped.
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # OpenCV's getRotationMatrix2D expects angle in degrees.
    # Positive angle for counter-clockwise rotation.
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # The output dimensions (w, h) for warpAffine will be the same as the input image.
    # Content outside these bounds is cropped.
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated