"""
Splits image files from a source directory into training and test sets.

This script provides functionality to:
- Discover image files (PNG, JPG, BMP) in a specified directory.
- Optionally randomize the order of these files.
- Split the files into training and test sets based on a specified training size.
- Move or copy the files into 'training' and 'test' subdirectories.
- Optionally convert images to PNG format.
- Optionally rotate images by 90 degrees clockwise during conversion.

Command-line arguments:
  --data_dir: Path to the source data directory containing the images.
  --training_size: The number of images to include in the training set.
  --make_test_dir: If specified, creates a 'test' directory for files not included in the training set.
  --make_random: If specified, randomizes the order of files before splitting.
  --convert_to_png: If specified, converts images to PNG format.
  --rotate_png_90: If specified (and --convert_to_png is also specified), rotates images 90 degrees clockwise.
  --move_files: If specified, moves files from the source directory to the training/test directories.
                Otherwise, files are copied.

Example usage:
  # Copy 100 random images to training, and the rest to test, converting to PNG
  python train_test_split.py --data_dir /path/to/images --training_size 100 --make_test_dir --make_random --convert_to_png

  # Move the first 50 images (in original order) to training, no test set, no conversion
  python train_test_split.py --data_dir /path/to/images --training_size 50 --move_files
"""
import glob
import random
import os
import random
import shutil
import cv2

def get_files(dir):
    """
    Retrieves a list of image files (PNG, JPG, BMP) from a directory.

    Args:
      dir (str): The directory to search for image files.

    Returns:
      list: A list of paths to the found image files.
    """
    ftypes=('*.png','*.jpg','*.bmp')
    files_grabbed = []
    for ftype in ftypes:
        files_grabbed.extend(glob.glob(os.path.join(dir,ftype)))
    print(f'[INFO] Found {len(files_grabbed)} available source files.')
    return files_grabbed

def randomize(files_list,seed=42):
    """
    Randomizes the order of elements in a list.

    Args:
      files_list (list): The list of files to randomize.
      seed (int, optional): The random seed to use for shuffling. Defaults to 42.

    Returns:
      list: The randomized list of files.
    """
    print(f'Randomizing source data.')
    random.seed(seed)
    random.shuffle(files_list)
    return files_list

def split_files(files_list,n,make_test):
    """
    Splits a list of files into training and test sets.

    Args:
      files_list (list): The list of files to split.
      n (int): The number of files to include in the training set.
      make_test (bool): If True, files not in the training set are added to the test set.

    Returns:
      tuple: A tuple containing two lists: (training_files, test_files).
             test_files will be empty if make_test is False or if n >= len(files_list).
    """
    training=files_list[0:n]
    test=[]
    if make_test:
        print('[INFO] Creating test directory for residual files not used in training set.')
        if len(files_list)>n:
            test=files_list[n:]
    return training,test

def move_files(dir,training,test,convert_to_png,rotate_png_90,make_test):
    """
    Moves files to 'training' and 'test' subdirectories, with optional PNG conversion and rotation.

    Args:
      dir (str): The base directory where 'training' and 'test' subdirectories will be created.
      training (list): A list of file paths for the training set.
      test (list): A list of file paths for the test set.
      convert_to_png (bool): If True, convert images to PNG format.
      rotate_png_90 (bool): If True (and convert_to_png is True), rotate images 90 degrees.
      make_test (bool): If True, process and move files in the 'test' list.
    """
    training_path=os.path.join(dir,'training/')
    test_path=os.path.join(dir,'test/')
    if os.path.exists(training_path):
        shutil.rmtree(training_path)
    os.makedirs(training_path)
    if make_test:
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        os.makedirs(test_path)

    for file in training:
        fname=os.path.split(file)[1]
        if convert_to_png:
            print(f'[INFO] Converting {fname} to .png and moving to training directory')
            img=cv2.imread(file)
            if rotate_png_90:
                img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            ext=os.path.splitext(fname)[1]
            fname_out=fname.replace(ext,'.png')
            path_out=os.path.join(training_path,fname_out)
            cv2.imwrite(path_out,img)
            print(f'[INFO] Removing original {fname} to .png')
            os.remove(file)
        else:
            print(f'[INFO] moving {fname} to training directory.')
            path_out=os.path.join(training_path,fname)
            shutil.move(file,path_out)
    if make_test:
        for file in test:
            fname=os.path.split(file)[1]
            if convert_to_png:
                print(f'[INFO] Converting {fname} to .png and moving to test dirctory')
                img=cv2.imread(file)
                ext=os.path.splitext(fname)[1]
                fname_out=fname.replace(ext,'.png')
                path_out=os.path.join(test_path,fname_out)
                cv2.imwrite(path_out,img)
                print(f'[INFO] Removing original {fname} to .png')
                os.remove(file)
            else:
                print(f'[INFO] moving {fname} to test directory.')
                path_out=os.path.join(test_path,fname)
                shutil.move(file,path_out)

def copy_files(dir,training,test,convert_to_png,rotate_png_90,make_test):
    """
    Copies files to 'training' and 'test' subdirectories, with optional PNG conversion and rotation.

    Args:
      dir (str): The base directory where 'training' and 'test' subdirectories will be created.
      training (list): A list of file paths for the training set.
      test (list): A list of file paths for the test set.
      convert_to_png (bool): If True, convert images to PNG format.
      rotate_png_90 (bool): If True (and convert_to_png is True), rotate images 90 degrees.
      make_test (bool): If True, process and copy files in the 'test' list.
    """
    training_path=os.path.join(dir,'training/')
    os.makedirs(training_path,exist_ok=True)
    if make_test:
        test_path=os.path.join(dir,'test/') 
        os.makedirs(test_path,exist_ok=True)
        
    print(f'[INFO] Copying {len(training)} files from {dir} to {training_path}')   
    for file in training:
        fname=os.path.split(file)[1]
        if convert_to_png:
            print(f'[INFO] Converting {fname} to .png and copying to training directory')
            img=cv2.imread(file)
            if rotate_png_90:
                img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            ext=os.path.splitext(fname)[1]
            fname_out=fname.replace(ext,'.png')
            path_out=os.path.join(training_path,fname_out)
            cv2.imwrite(path_out,img)
        else:
            print(f'[INFO] Copying {fname} to training directory.')
            path_out=os.path.join(training_path,fname)
            shutil.copy(file,path_out)
    if make_test:
        for file in test:
            fname=os.path.split(file)[1]
            if convert_to_png:
                print(f'[INFO] Converting {fname} to .png and moving to test dirctory')
                img=cv2.imread(file)
                ext=os.path.splitext(fname)[1]
                fname_out=fname.replace(ext,'.png')
                path_out=os.path.join(test_path,fname_out)
                cv2.imwrite(path_out,img)
            else:
                print(f'[INFO] Copying {fname} to test directory.')
                path_out=os.path.join(test_path,fname)
                shutil.copy(file,path_out)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Splits image files from a source directory into training and test sets.")
    parser.add_argument("--data_dir", required=True, help='Source data directory.  All training data will be moved/copied to data_dir/training dir.')
    parser.add_argument("--training_size",type=int,default=0, help="Training data size.")
    parser.add_argument("--make_test_dir", action='store_true',help='Set to create a test directory. All non-training files will be moved/copied to data_dir/test dir.  Default to no test dir.')
    parser.add_argument("--make_random", action='store_true',help='Randomize training data. Default to no randomizing.')
    parser.add_argument("--convert_to_png", action='store_true',help='Convert imags to .png. Default to not image recasting.')
    parser.add_argument("--rotate_png_90",action='store_true',help='Rotate images 90 degrees. Default to no rotation.')
    parser.add_argument("--move_files",action='store_true',help='Move files deletes original files. Default to copy.')
    args = parser.parse_args()

    files_list=get_files(args.data_dir)
    if args.make_random:
        files_list=randomize(files_list)
    training_list,test_list=split_files(files_list,args.training_size,args.make_test_dir)

    if args.move_files:
        move_files(args.data_dir,training_list,test_list,args.convert_to_png,args.rotate_png_90,args.make_test_dir)
    else:
        copy_files(args.data_dir,training_list,test_list,args.convert_to_png,args.rotate_png_90,args.make_test_dir)