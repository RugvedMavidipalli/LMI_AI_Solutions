"""
Provides utilities for converting 2D image data stored in LMI's Gadget format
(specifically '.gadget2d.pickle' files) to and from common image formats like
NumPy arrays (.npy) and PNG images (.png).

The script can be used as a command-line tool to perform batch conversions
on directories of files.

Command-line arguments:
  --option: The conversion operation to perform.
            Choices: 'pkl_2_npy', 'pkl_2_png', 'npy_2_pkl', 'png_2_pkl'. (required)
  --src:    Path to the source directory containing files to convert. (required)
  --dest:   Path to the destination directory where converted files will be saved.
            If it doesn't exist, it will be created. (required)
  --rotate: If specified, rotates the image data by 90 degrees clockwise during conversion.
            (optional, default: False)

Example usage (command-line):
  # Convert .gadget2d.pickle files in 'input_pickles' to .npy files in 'output_npys'
  python gadget_image_utils.py --option pkl_2_npy --src input_pickles --dest output_npys

  # Convert .npy files in 'input_npys' to .gadget2d.pickle files in 'output_pickles', with rotation
  python gadget_image_utils.py --option npy_2_pkl --src input_npys --dest output_pickles --rotate
"""
from PIL import Image
import pickle
import numpy
from os import listdir, makedirs
from os.path import isfile, join, isdir

class GadgetImageUtils():
    """
    A utility class for converting LMI Gadget 2D image formats.

    This class handles conversions between '.gadget2d.pickle' files,
    NumPy arrays, and PNG image files. It defines a schema ID and version
    for the Gadget format it produces.
    """

    SCHEMA_ID: str = "gadget2d" 
    VERSION: int = 1

    def pkl_2_npy(self, source_path: str, destination_path: str, rotate: bool = False):
        """
        Converts .gadget2d.pickle files to NumPy .npy files.

        Args:
            source_path (str): Directory containing .gadget2d.pickle files.
            destination_path (str): Directory to save the converted .npy files.
            rotate (bool, optional): If True, rotates the image 90 degrees clockwise.
                                     Defaults to False.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget2d.pickle" in f]

        for file in files:
            print(join(source_path, file))
            
            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)
            npy_arr = content["pixel_array"]

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            numpy.save(join(destination_path, file.replace('.gadget2d.pickle', '.npy')), npy_arr)

    def pkl_2_png(self, source_path: str, destination_path: str, rotate: bool = False):
        """
        Converts .gadget2d.pickle files to PNG image files.

        Args:
            source_path (str): Directory containing .gadget2d.pickle files.
            destination_path (str): Directory to save the converted .png files.
            rotate (bool, optional): If True, rotates the image 90 degrees clockwise.
                                     Defaults to False.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget2d.pickle" in f]

        for file in files:
            print(join(source_path, file))
            
            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)
            
            npy_arr = content["pixel_array"]

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            image = Image.fromarray(npy_arr)
            image.save(join(destination_path, file.replace('.gadget2d.pickle', '.png')))
    
    def npy_2_pkl(self, source_path: str, destination_path: str, rotate: bool = False):
        """
        Converts NumPy .npy files to .gadget2d.pickle files.

        The pixel format in the output pickle ('RGB_8' or 'GRAY_8') is inferred
        from the shape of the NumPy array.

        Args:
            source_path (str): Directory containing .npy files.
            destination_path (str): Directory to save the converted .gadget2d.pickle files.
            rotate (bool, optional): If True, rotates the image 90 degrees clockwise
                                     before saving. Defaults to False.
        Raises:
            ValueError: If the input NumPy array has an unsupported shape.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".npy" in f]

        for file in files:
            print(join(source_path, file))

            npy_arr = numpy.load(join(source_path, file))

            if npy_arr.ndim == 3 and npy_arr.shape[2] == 3:
                pixel_format = "RGB_8"
            elif npy_arr.ndim == 2:
                pixel_format = "GRAY_8"
            else:
                raise ValueError(f"Unsupported array shape: {npy_arr.shape} for file {file}")

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "pixel_format": pixel_format
                }, 
                "pixel_array": npy_arr,
            }
            
            with open(join(destination_path, file.replace('.npy', '.gadget2d.pickle')), "wb") as f:
                pickle.dump(content, f, protocol=4)

    def png_2_pkl(self, source_path: str, destination_path: str, rotate: bool = False):
        """
        Converts PNG image files to .gadget2d.pickle files.

        The pixel format in the output pickle ('RGB_8' or 'GRAY_8') is inferred
        from the mode of the PNG image (e.g., 'RGB' or 'L').

        Args:
            source_path (str): Directory containing .png files.
            destination_path (str): Directory to save the converted .gadget2d.pickle files.
            rotate (bool, optional): If True, rotates the image 90 degrees clockwise
                                     before saving. Defaults to False.
        Raises:
            ValueError: If the input PNG image has an unsupported array shape after conversion.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".png" in f]

        for file in files:
            print(join(source_path, file))

            img = Image.open(join(source_path, file))
            npy_arr = numpy.array(img)

            if npy_arr.ndim == 3 and npy_arr.shape[2] == 3:
                pixel_format = "RGB_8"
            elif npy_arr.ndim == 2:
                pixel_format = "GRAY_8"
            # Handling PNGs with alpha channel (e.g. RGBA) more explicitly if needed:
            # elif npy_arr.ndim == 3 and npy_arr.shape[2] == 4:
            #     pixel_format = "RGBA_8" # Or convert to RGB_8 by dropping alpha
            #     npy_arr = npy_arr[:,:,:3] # Example: Convert RGBA to RGB
            else:
                raise ValueError(f"Unsupported array shape: {npy_arr.shape} from PNG file {file}. Mode: {img.mode}")

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "pixel_format": pixel_format
                }, 
                "pixel_array": npy_arr,
            }
            
            with open(join(destination_path, file.replace('.png', '.gadget2d.pickle')), "wb") as f:
                pickle.dump(content, f, protocol=4)


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(
        description="Convert 2D image data between LMI Gadget format (.gadget2d.pickle), "
                    "NumPy arrays (.npy), and PNG images (.png)."
    )
    ap.add_argument('--option',required=True,
                        choices=['pkl_2_npy', 'pkl_2_png', 'npy_2_pkl', 'png_2_pkl'],
                        help='The conversion operation to perform.')
    ap.add_argument('--src',required=True, help='Path to the source directory.')
    ap.add_argument('--dest',required=True, help='Path to the destination directory. Will be created if it does not exist.')
    ap.add_argument('--rotate', action='store_true',help='Rotate the image 90 degrees clockwise during conversion.')
    
    args=vars(ap.parse_args())
    option=args['option']
    src_path=args['src'] # Renamed for clarity
    dest_path=args['dest'] # Renamed for clarity
    rotate_flag = args['rotate'] # Renamed for clarity

    print(f"Operation: {option}")
    print(f"Source: {src_path}")
    print(f"Destination: {dest_path}")
    print(f"Rotate: {rotate_flag}")
    
    converter = GadgetImageUtils() # Renamed for clarity

    if not isdir(dest_path):
        print(f"Creating destination directory: {dest_path}")
        makedirs(dest_path)

    if option=='pkl_2_npy':
        converter.pkl_2_npy(src_path, dest_path, rotate_flag)
    elif option=='pkl_2_png':
        converter.pkl_2_png(src_path, dest_path, rotate_flag)
    elif option=='npy_2_pkl':
        converter.npy_2_pkl(src_path, dest_path, rotate_flag)
    elif option=='png_2_pkl':
        converter.png_2_pkl(src_path, dest_path, rotate_flag)
    else:
        # This case should not be reached due to argparse choices
        raise Exception('Invalid option. Supported options are: pkl_2_npy, pkl_2_png, npy_2_pkl, png_2_pkl')

    print("Conversion complete.")