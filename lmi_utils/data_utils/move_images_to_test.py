"""
Moves images from a source directory to a destination directory based on a list of filenames.

This script is useful for organizing image datasets, for example, by moving a specific set of images
to a 'test' or 'validation' folder.

Command-line arguments:
  -s, --src: Path to the source directory. This can be a parent directory containing multiple
             subdirectories with images.
  -d, --dest: Path to the destination directory. If it doesn't exist, it will be created.
  -f, --file: Path to a file containing a list of image filenames to be moved. Each filename
              should be on a new line.

Example usage:
  python move_images_to_test.py --src /path/to/source_images --dest /path/to/test_images --file /path/to/images_to_move.txt
"""
import shutil
import os
import argparse
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Moves images from a source directory to a destination directory based on a list of filenames.")
    parser.add_argument('-s', '--src', required=True, help='path to source directory. This could be a parent directory of all images, such as each-sku')
    parser.add_argument('-d', '--dest', required=True, help='path to destination directory')
    parser.add_argument('-f', '--file', required=True, help='path to file with a list of images to be moved')
    
    args = parser.parse_args()
    
    to_be_moved = set()
    with open(args.file, 'r') as f:
        for line in f:
            to_be_moved.add(line.strip())
    logger.info(f'{len(to_be_moved)} images to be removed')
    
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    for root, dirs, files in os.walk(args.src):
        for file in files:
            if file in to_be_moved:
                path = os.path.join(root, file)
                
                # skip if already exists
                if os.path.isfile(os.path.join(args.dest, file)):
                    logger.info(f'{file} already exists in {args.dest}, skip')
                    continue
                
                logger.info(f'move {path} to {args.dest}')
                shutil.move(path, args.dest)
