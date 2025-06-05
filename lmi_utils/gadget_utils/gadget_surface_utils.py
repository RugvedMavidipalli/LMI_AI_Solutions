"""
Provides utilities for converting 3D surface data stored in LMI's Gadget format
(specifically '.gadget3d.pickle' files) to and from other common 3D and 2D formats
like NumPy arrays (.npy), PNG images (.png for profile or intensity), and
Point Cloud Data (.pcd) files.

The script can be used as a command-line tool to perform batch conversions
on directories of files. It handles profile data and optionally intensity data.
For conversions involving .pcd files, the Open3D library is required.
For conversions involving .tar files (containing profile and metadata), the tarfile
and json libraries are used.

Command-line arguments:
  --option: The conversion operation to perform. (required)
            Choices: 'pkl_2_npy', 'pkl_2_png', 'pkl_2_pcd', 'npy_2_pkl',
                     'png_2_pkl', 'tar_2_pcd', 'pcd_2_pkl'.
  --src:    Path to the source directory containing files to convert. (required)
  --dest:   Path to the destination directory where converted files will be saved.
            If it doesn't exist, it will be created. (required)
  --src_intensity: Path to a directory containing intensity images (e.g., .gadget2d.jpg
                   or .png) corresponding to the source 3D data. Used for operations
                   like 'tar_2_pcd' or 'png_2_pkl' if separate intensity is provided. (optional)
  --intensity: If specified when converting from .pkl, also saves the intensity array
               if present in the pickle file (as a separate -intensity.npy or -intensity.png).
               (optional, default: False)
  --zresolution: Z-axis resolution value, required for 'pcd_2_pkl' conversion. (optional)
  --zoffset:     Z-axis offset value, required for 'pcd_2_pkl' conversion. (optional)


Example usage (command-line):
  # Convert .gadget3d.pickle files in 'input_pickles' to .npy profile arrays (and intensity if --intensity is used)
  python gadget_surface_utils.py --option pkl_2_npy --src input_pickles --dest output_npys --intensity

  # Convert .gadget3d.tar files to .pcd files, using intensity from a separate directory
  python gadget_surface_utils.py --option tar_2_pcd --src input_tars --dest output_pcds --src_intensity intensity_images_dir

  # Convert .pcd files to .gadget3d.pickle files with specified Z resolution and offset
  python gadget_surface_utils.py --option pcd_2_pkl --src input_pcds --dest output_pickles --zresolution 0.01 --zoffset 100.0
"""
from PIL import Image
import pickle
import numpy
from os import listdir, makedirs
from os.path import isfile, join, isdir
import cv2

TWO_TO_FIFTEEN=32768

class GadgetSurfaceUtils():
    """
    A utility class for converting LMI Gadget 3D surface formats.

    This class handles conversions between '.gadget3d.pickle' files (containing
    profile and optionally intensity data), NumPy arrays, PNG images, and
    Point Cloud Data (.pcd) files. It defines a schema ID and version
    for the Gadget format it produces.
    """

    SCHEMA_ID: str = "gadget3d"
    VERSION: int = 1

    def pkl_2_npy(self, source_path: str, destination_path: str, intensity: bool = False):
        """
        Converts .gadget3d.pickle files to NumPy .npy files.

        Extracts the 'profile_array' and optionally the 'intensity_array'.
        Profile data (int16) is converted to uint16 by adding 32768 before saving as .npy.

        Args:
            source_path (str): Directory containing .gadget3d.pickle files.
            destination_path (str): Directory to save the converted .npy files.
                                    Profile array is saved as <filename>.npy.
                                    Intensity array (if present and `intensity` is True)
                                    is saved as <filename>-intensity.npy.
            intensity (bool, optional): If True, also extracts and saves the
                                        intensity array if it exists. Defaults to False.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)

            profile = content["profile_array"]
            if profile.dtype == numpy.int16:
                profile = profile.view(numpy.uint16) + numpy.uint16(32768)
            
            numpy.save(join(destination_path, file.replace('.gadget3d.pickle', '.npy')), profile)

            if intensity:
                try:
                    if content["intensity_array"] is not None:
                        numpy.save(join(destination_path, file.replace('.gadget3d.pickle', '-intensity.npy')), content["intensity_array"])
                except KeyError:
                    continue
            
    def pkl_2_png(self, source_path: str, destination_path: str, intensity: bool = False):
        """
        Converts .gadget3d.pickle files to PNG image files.

        Extracts the 'profile_array' and optionally the 'intensity_array'.
        Profile data (int16) is converted to uint16 by adding 32768 before saving as PNG.
        The arrays are saved as grayscale PNG images.

        Args:
            source_path (str): Directory containing .gadget3d.pickle files.
            destination_path (str): Directory to save the converted .png files.
                                    Profile array is saved as <filename>.png.
                                    Intensity array (if present and `intensity` is True)
                                    is saved as <filename>-intensity.png.
            intensity (bool, optional): If True, also extracts and saves the
                                        intensity array if it exists. Defaults to False.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)

            profile = content["profile_array"]
            if profile.dtype == numpy.int16:
                profile = profile.view(numpy.uint16) + numpy.uint16(32768)
            
            image = Image.fromarray(profile)
            image.save(join(destination_path, file.replace('.gadget3d.pickle', '.png')))
            
            if intensity:
                try:
                    if content["intensity_array"] is not None:
                        image = Image.fromarray(content["intensity_array"])
                        image.save(join(destination_path, file.replace('.gadget3d.pickle', '-intensity.png')))
                except KeyError:
                    continue
                
           
    @staticmethod     
    def convert_to_xyz(profile: numpy.ndarray, resolution: tuple, offset: tuple, img_intensity: numpy.ndarray = None) -> tuple:
        """
        Converts a 2D profile array (height map) into a 3D point cloud (X, Y, Z coordinates).

        Optionally includes intensity data for each point if an intensity image is provided.
        Assumes that -32768 in the input profile represents invalid points (no data).

        Args:
            profile (numpy.ndarray): 2D array representing the height map.
                                     Expected to be int16 or uint16. If uint16, it's
                                     converted to int16 by subtracting 32768.
            resolution (tuple): A tuple (x_res, y_res, z_res) representing the
                                resolution in each dimension.
            offset (tuple): A tuple (x_off, y_off, z_off) representing the
                            offset in each dimension.
            img_intensity (numpy.ndarray, optional): A 2D array of the same shape as
                                                     `profile`, containing intensity values.
                                                     These are normalized to [0,1].
                                                     Defaults to None.

        Returns:
            tuple:
                - np_points (numpy.ndarray): An Nx3 array of (X, Y, Z) coordinates.
                - np_intensity (numpy.ndarray): An Nx1 array of intensity values,
                                                or an empty array if `img_intensity` was None.
        """
        np_z=[]
        np_x=[]
        np_y=[]
        intensity=[]
        i = 0
        
        # convert to int16
        # the metadata only works with int16
        if profile.dtype == numpy.uint16:
            profile = profile.view(numpy.int16) + numpy.int16(-TWO_TO_FIFTEEN)

        shape = profile.shape
        for y in range(shape[0]):
            for x in range(shape[1]):
                if profile[y][x] != -TWO_TO_FIFTEEN:
                    np_x.append(offset[0] + x * resolution[0])
                    np_y.append(offset[1] + y * resolution[1])
                    np_z.append(offset[2] + profile[y][x] * resolution[2])
                    if img_intensity is not None:
                        intensity.append(img_intensity[y][x]/255.0)
                    i += 1
        
        np_points = numpy.empty((i, 3))
        np_points[:, 0] = numpy.array(np_x)
        np_points[:, 1] = numpy.array(np_y)
        np_points[:, 2] = numpy.array(np_z)
        np_intensity=numpy.array(intensity)
        return np_points,np_intensity
        
                
    def pkl_2_pcd(self, source_path: str, destination_path: str, source_path_intensity: str = None):
        """
        Converts .gadget3d.pickle files to .pcd (Point Cloud Data) files.

        Requires the Open3D library.
        If `source_path_intensity` is provided, it attempts to load corresponding
        '.gadget2d.jpg' intensity images and include their data as colors in the PCD.

        Args:
            source_path (str): Directory containing .gadget3d.pickle files.
            destination_path (str): Directory to save the converted .pcd files.
            source_path_intensity (str, optional): Directory containing corresponding
                                                   intensity images (e.g., gadget2d.jpg).
                                                   Defaults to None.
        """
        import open3d
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]
        
        use_intensity=True if source_path_intensity is not None else False

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)
            
            if use_intensity:
                try:
                    fname_intensity=file.replace(".gadget3d.pickle", ".gadget2d.jpg")
                    path_intensity=join(source_path_intensity,fname_intensity)
                    print(f'[INFO] Loading intensity image from:{path_intensity}')
                    img_intensity=Image.open(path_intensity)
                    img_intensity=img_intensity.convert('RGB') #convert to color
                    img_intensity=numpy.array(img_intensity).astype(numpy.float32)
                except:
                    print(f'[WARNING] Failed to load intensity image.')
                    use_intensity=False

            profile = content["profile_array"]
            resolution = content["metadata"]["resolution"]
            offset = content["metadata"]["offset"]

            # convert to 3d points
            np_points,np_intensity = self.convert_to_xyz(profile, resolution, offset, img_intensity if use_intensity else None)
            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np_points)
            if use_intensity:
                pcd.colors = open3d.utility.Vector3dVector(np_intensity)
            open3d.io.write_point_cloud(join(destination_path, file.replace(".gadget3d.pickle", ".pcd")), pcd)
    
    def tar_2_pcd(self, source_path: str, destination_path: str, source_path_intensity: str = None):
        """
        Converts .gadget3d.tar archive files to .pcd (Point Cloud Data) files.

        Each .tar file is expected to contain 'profile.png' and 'metadata.json'.
        Requires Open3D, tarfile, and json libraries.
        If `source_path_intensity` is provided, it attempts to load corresponding
        '.gadget2d.jpg' intensity images and include their data as colors in the PCD.

        Args:
            source_path (str): Directory containing .gadget3d.tar files.
            destination_path (str): Directory to save the converted .pcd files.
                                    A subdirectory for each tar file's contents will
                                    also be created within `destination_path`.
            source_path_intensity (str, optional): Directory containing corresponding
                                                   intensity images (e.g., gadget2d.jpg).
                                                   Defaults to None.
        """
        import open3d
        import tarfile
        import json

        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.tar" in f]

        use_intensity=True if source_path_intensity is not None else False
        
        for file in files:
            print(join(source_path, file))
            
            with tarfile.open(join(source_path, file), "r") as tar:
                dest = join(destination_path  , file.replace(".gadget3d.tar", ""))
                tar.extractall(dest)

                png = cv2.imread(join(dest, "profile.png"), cv2.IMREAD_UNCHANGED)
                if use_intensity:
                    try:
                        fname_intensity=file.replace(".gadget3d.tar", ".gadget2d.jpg")
                        path_intensity=join(source_path_intensity,fname_intensity)
                        print(f'[INFO] Loading intensity image from:{path_intensity}')
                        img_intensity=Image.open(path_intensity)
                        img_intensity=img_intensity.convert('RGB') #convert to color
                        img_intensity=numpy.array(img_intensity).astype(numpy.float32)
                    except:
                        print(f'[WARNING] Failed to load intensity image.')
                        use_intensity=False
                        
                metadata = None
                with open(join(dest, "metadata.json"), "r") as f:
                    metadata = json.load(f)

                profile = numpy.array(png)
                resolution = metadata["resolution"]
                offset = metadata["offset"]

                # convert to 3d points
                np_points,np_intensity = self.convert_to_xyz(profile, resolution, offset, img_intensity if use_intensity else None)
                
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np_points)
                if use_intensity:
                    pcd.colors = open3d.utility.Vector3dVector(np_intensity)
                open3d.io.write_point_cloud(join(destination_path, file.replace(".gadget3d.tar", ".pcd")), pcd)


    def npy_2_pkl(self, source_path: str, destination_path: str):
        """
        Converts NumPy .npy files (profile arrays) to .gadget3d.pickle files.

        Assumes the .npy file contains a 2D profile array.
        Handles dtype conversion from uint16 (as saved by pkl_2_npy) back to int16
        by subtracting 32768. Also handles int32, assuming it needs similar offset.
        Metadata for resolution and offset are set to default values (1 and 0).
        Intensity array is set to None.

        Args:
            source_path (str): Directory containing .npy profile files.
            destination_path (str): Directory to save the converted .gadget3d.pickle files.
        """
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".npy" in f]

        for file in files:
            print(join(source_path, file))

            npy_arr = numpy.load(join(source_path, file))
            
            if npy_arr.dtype == numpy.uint16:
                npy_arr = npy_arr.view(numpy.int16) + numpy.int16(-32768) 
            elif npy_arr.dtype == numpy.int32:
                npy_arr = (npy_arr - 32768).astype(numpy.int16) 
            
            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "resolution": 1, 
                    "offset": 0, 
                }, 
                "profile_array": npy_arr,
                "intensity_array": None,
            }

            with open(join(destination_path, file.replace(".npy", ".gadget3d.pickle")), "wb") as f: # Corrected .png to .npy
                pickle.dump(content, f, protocol=4)

    def png_2_pkl(self, source_path: str, destination_path: str, source_path_intensity: str = None):
        """
        Converts PNG image files (profile and optional intensity) to .gadget3d.pickle files.

        Profile PNGs are read from `source_path`. If `source_path_intensity` is provided,
        correspondingly named PNGs are read from there for intensity data.
        Handles dtype conversion for profile data similarly to `npy_2_pkl`.
        Metadata for resolution and offset are set to default values ((1,1,1) and (0,0,0)).

        Args:
            source_path (str): Directory containing profile .png files.
            destination_path (str): Directory to save the converted .gadget3d.pickle files.
            source_path_intensity (str, optional): Directory containing intensity .png files.
                                                   Filenames should correspond to profile PNGs.
                                                   Defaults to None.
        """
        files_p = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".png" in f]
        files_p.sort()
        if source_path_intensity is not None:
            files_i = [f for f in listdir(source_path_intensity) if isfile(join(source_path_intensity, f)) and ".png" in f]
            files_i.sort()
            files=zip(files_p,files_i)
        else:
            files=zip(files_p)
        
        for file_tuple in files:
            file_p=file_tuple[0]
            print(join(source_path, file_p))
            img = Image.open(join(source_path, file_p))
            npy_arr_p = numpy.array(img)
            if npy_arr_p.dtype == numpy.uint16:
                npy_arr_p = npy_arr_p.view(numpy.int16) + numpy.int16(-32768) 
            elif npy_arr_p.dtype == numpy.int32:
                npy_arr_p = (npy_arr_p - 32768).astype(numpy.int16) 
            
            if len(file_tuple)==2:
                file_i=file_tuple[1]
                print(join(source_path, file_i))
                img = Image.open(join(source_path_intensity, file_i))
                npy_arr_i = numpy.array(img)
            else:
                npy_arr_i=None

            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "resolution": (1,1,1), 
                    "offset": (0,0,0), 
                }, 
                "profile_array": npy_arr_p,
                "intensity_array": npy_arr_i,
            }
               
            with open(join(destination_path, file_p.replace(".png", ".gadget3d.pickle")), "wb") as f:
                pickle.dump(content, f, protocol=4)

    def pcd_2_pkl(self, source_path: str, destination_path: str, ZResolution: float = 1.0, ZOffset: float = 0.0):
        """
        Converts .pcd (Point Cloud Data) files to .gadget3d.pickle files.

        Requires the Open3D library.
        The method attempts to reconstruct a 2D profile array from the point cloud.
        It infers X and Y resolution and offset from the point data.
        Z resolution and offset must be provided. Intensity data is set to None.

        Args:
            source_path (str): Directory containing .pcd files.
            destination_path (str): Directory to save the converted .gadget3d.pickle files.
            ZResolution (float, optional): Z-axis resolution for scaling Z values.
                                           Defaults to 1.0.
            ZOffset (float, optional): Z-axis offset for Z values. Defaults to 0.0.
        """
        import open3d
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".pcd" in f]

        for file in files:
            print(join(source_path, file))

            pcd = open3d.io.read_point_cloud(join(source_path, file))
            np_arr = numpy.asarray(pcd.points)

            x_len = 0
            for np in np_arr:
                if np[1] != np_arr[0][1]:
                    break
                x_len += 1

            y_len = int(np_arr.shape[0] / x_len)
            
            XOffset = float(np_arr[0][0])
            XResolution = float((np_arr[1][0] - np_arr[0][0]))
            YOffset = float(np_arr[0][1])
            YResolution = float((np_arr[x_len][1] - np_arr[0][1]))

            np_z = numpy.empty((x_len, y_len))
            i = 0
            for y in range(0, y_len):
                for x in range(0, x_len):
                    np_z[x][y] = (np_arr[i][2] - ZOffset) / ZResolution
                    i += 1
                
            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "resolution": (XResolution, YResolution, ZResolution), 
                    "offset": (XOffset, YOffset, ZOffset), 
                }, 
                "profile_array": np_z.astype(numpy.int16),
                "intensity_array": None,
            }

            with open(join(destination_path, file.replace(".pcd", ".gadget3d.pickle")), "wb") as f:
                pickle.dump(content, f, protocol=4)



if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Convert 3D surface data between LMI Gadget format (.gadget3d.pickle), "
                    "NumPy arrays (.npy), PNG images (.png), PCD files (.pcd), and TAR archives (.gadget3d.tar)."
    )
    ap.add_argument('--option', required=True,
                        choices=['pkl_2_npy', 'pkl_2_png', 'pkl_2_pcd', 'npy_2_pkl',
                                 'png_2_pkl', 'tar_2_pcd', 'pcd_2_pkl'],
                        help='The conversion operation to perform.')
    ap.add_argument('--src', required=True, help='Path to the source directory.')
    ap.add_argument('--dest', required=True, help='Path to the destination directory. Will be created if it does not exist.')
    ap.add_argument('--src_intensity', default=None,
                        help='Path to a directory containing intensity images (e.g., .gadget2d.jpg or .png) '
                             'corresponding to the source 3D data. Used for operations like "tar_2_pcd" or '
                             '"png_2_pkl" if separate intensity is provided.')
    ap.add_argument('--intensity', action='store_true',
                        help='If specified when converting from .pkl, also saves the intensity array '
                             'if present in the pickle file (as a separate -intensity.npy or -intensity.png).')
    ap.add_argument('--zresolution', type=float, help='Z-axis resolution value, required for "pcd_2_pkl" conversion.')
    ap.add_argument('--zoffset', type=float, help='Z-axis offset value, required for "pcd_2_pkl" conversion.')

    
    args = ap.parse_args() # Use parse_args() directly
    option = args.option
    src_path = args.src
    dest_path = args.dest
    src_intensity_path = args.src_intensity # Renamed for clarity
    intensity_flag = args.intensity # Renamed for clarity

    converter = GadgetSurfaceUtils() # Renamed for clarity

    print(f"Operation: {option}")
    print(f"Source: {src_path}")
    print(f"Destination: {dest_path}")
    if src_intensity_path:
        print(f"Source Intensity: {src_intensity_path}")
    if option in ['pkl_2_npy', 'pkl_2_png']:
        print(f"Include Intensity output: {intensity_flag}")
    
    if not isdir(dest_path):
        print(f"Creating destination directory: {dest_path}")
        makedirs(dest_path)

    if option == 'pkl_2_npy':
        converter.pkl_2_npy(src_path, dest_path, intensity_flag)
    elif option == 'pkl_2_png':
        converter.pkl_2_png(src_path, dest_path, intensity_flag)
    elif option == 'pkl_2_pcd':
        # Assuming pkl_2_pcd might use src_intensity_path if provided for consistency,
        # or it might only use intensity from within the pickle. The method itself handles it.
        converter.pkl_2_pcd(src_path, dest_path, source_path_intensity=src_intensity_path)
    elif option == 'npy_2_pkl':
        converter.npy_2_pkl(src_path, dest_path)
    elif option == 'png_2_pkl':
        converter.png_2_pkl(src_path, dest_path, source_path_intensity=src_intensity_path)
    elif option == 'tar_2_pcd':
        converter.tar_2_pcd(src_path, dest_path, source_path_intensity=src_intensity_path)
    elif option == 'pcd_2_pkl':
        if args.zresolution is None or args.zoffset is None:
            raise ValueError("--zresolution and --zoffset are required for pcd_2_pkl conversion.")
        converter.pcd_2_pkl(src_path, dest_path, args.zresolution, args.zoffset)
    else:
        # This case should not be reached due to argparse choices
        raise Exception(f"Invalid option '{option}'. Check --help.")

    print("Conversion complete.")
