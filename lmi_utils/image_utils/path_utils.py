"""
Provides utility functions for path manipulation, primarily for finding image files
within directories.

This module contains a function `get_relative_paths` that scans a given directory
(optionally recursively) for image files with specified formats and returns
their paths relative to the input directory.

The script can be run from the command line to test the `get_relative_paths`
function by printing the relative paths of images found in a specified directory.

Command-line arguments (for testing):
  -i, --input_path: Path to the directory to scan for images. (required)
  --recursive: If specified, searches recursively through subdirectories.
               (optional, default: False)

Example usage (command-line for testing):
  # List all .png, .jpg, .jpeg, .tiff images in 'my_dataset' and its subfolders
  python path_utils.py -i my_dataset --recursive
"""
import os
import logging
from typing import List # For type hinting

logging.basicConfig() # Consider moving this to the main block if used only by CLI test
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IMG_FORMATS: List[str] = ['.png','.jpg','.jpeg','.tiff']


def get_relative_paths(inpath: str, recursive: bool = True, formats: List[str] = IMG_FORMATS) -> List[str]:
    """
    Gets the relative paths of image files within a given input directory.

    Args:
        inpath (str): The path to the input directory.
        recursive (bool, optional): If True, searches recursively through subdirectories.
                                    Defaults to True.
        formats (List[str], optional): A list of file extensions (including the dot)
                                       to consider as image formats.
                                       Defaults to `IMG_FORMATS` global variable
                                       ( ['.png','.jpg','jpeg','tiff'] ).

    Returns:
        List[str]: A list of relative file paths to the found image files, with respect
                   to `inpath`.

    Raises:
        TypeError: If `formats` is not a list.
        FileNotFoundError: If `inpath` does not exist or is not a directory.
    """
    if not isinstance(formats, list):
        raise TypeError(f'formats must be a list of strings. But got the type: {type(formats)}')
    if not os.path.isdir(inpath):
        raise FileNotFoundError(f"Input path '{inpath}' is not a valid directory or does not exist.")
    
    logger.info(f'Searching for files with extensions: {formats} in directory: {inpath}')
    files = []
    for root, dirs, fs in os.walk(inpath):
        cnt = 0
        for file in fs:
            if os.path.splitext(file)[1] in formats:
                files.append(os.path.relpath(os.path.join(root, file), inpath))
                cnt += 1
        logger.info(f'Load {cnt} files in {root}')
        
        if not recursive:
            break
    return files
    
    
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Lists relative paths of image files in a directory."
    )
    ap.add_argument('-i','--input_path', required=True, help='Path to the directory to scan for images.')
    ap.add_argument('--recursive', action='store_true', help='If set, searches recursively through subdirectories.')
    args = ap.parse_args()
    
    try:
        paths = get_relative_paths(args.input_path, args.recursive)
    except (TypeError, FileNotFoundError) as e:
        logger.error(e)
        paths = []

    for p in paths:
        logger.info(f'output path: {p}')