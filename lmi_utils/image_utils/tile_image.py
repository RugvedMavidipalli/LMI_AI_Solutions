"""
Provides utilities for tiling large images into smaller, potentially overlapping,
tiles and for reconstructing the original image from these tiles.

When tiling, the script first resizes the source image to ensure its dimensions
are compatible with the specified tile size and stride. It then extracts tiles
and saves them individually. Metadata about the tiling process (original dimensions,
tile size, stride) is saved in a JSON file, which is crucial for reconstruction.

When reconstructing, the script reads the tiles and the metadata JSON from a source
directory and stitches the tiles back together to form the original image.
It supports different modes for handling overlapping regions during reconstruction
(e.g., 'max' or 'avg' intensity).

Command-line arguments:
  --option: The operation to perform. Choices: 'tile', 'reconstruct'. (required)
  --src:    Path to the source.
            For 'tile': path to the input image file or directory of images.
            For 'reconstruct': path to the directory containing tiles and metadata.json. (required)
  --dest:   Path to the destination directory.
            For 'tile': directory where tiles and metadata.json will be saved.
            For 'reconstruct': directory where reconstructed image(s) will be saved. (required)
  --w:      Tile width in pixels. (default: 224, for 'tile' option)
  --h:      Tile height in pixels. (default: 224, for 'tile' option)
  --stride_w: Stride width for tiling. If None, defaults to tile width (no overlap).
              (optional, for 'tile' option)
  --stride_h: Stride height for tiling. If None, defaults to tile height (no overlap).
              (optional, for 'tile' option)
  --overlap_mode: Mode for handling overlapping regions during reconstruction.
                  Choices: 'max', 'avg'. (default: 'max', for 'reconstruct' option)
  --type:   Image type/extension (e.g., 'png', 'jpg') to process when tiling a directory.
            (default: 'png', for 'tile' option with directory source)

Example usage (command-line):
  # Tile a large image 'big_image.png' into 512x512 tiles with 256px stride
  python tile_image.py --option tile --src big_image.png --dest output_tiles --w 512 --h 512 --stride_w 256 --stride_h 256

  # Reconstruct image(s) from tiles in 'output_tiles', save to 'reconstructed_images'
  python tile_image.py --option reconstruct --src output_tiles --dest reconstructed_images --overlap_mode avg

  # Tile all 'jpg' images in 'input_folder'
  python tile_image.py --option tile --src input_folder --dest output_tiles_jpg --type jpg
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import json
from typing import Optional # For type hinting


def __tile_image(source: Path, dest: Path, tile_w: int, tile_h: int, stride_w: Optional[int] = None, stride_h: Optional[int] = None) -> None:
    """
    Internal helper function to tile a single image.

    Resizes the image to ensure its dimensions are multiples of the stride plus
    the tile dimension, then extracts and saves tiles. Also saves metadata.

    Args:
        source (Path): Path to the source image file.
        dest (Path): Path to the destination directory for tiles and metadata.
        tile_w (int): Width of each tile.
        tile_h (int): Height of each tile.
        stride_w (Optional[int]): Stride width. Defaults to `tile_w` if None.
        stride_h (Optional[int]): Stride height. Defaults to `tile_h` if None.
    """
    try:
        with Image.open(source) as img:
            original_width, original_height = img.size

            _stride_w = stride_w if stride_w is not None else tile_w
            _stride_h = stride_h if stride_h is not None else tile_h

            if _stride_w <= 0 or _stride_h <= 0:
                logging.error("Strides must be positive. Skipping tiling for %s.", source)
                return
            if tile_w <= 0 or tile_h <= 0:
                logging.error("Tile dimensions must be positive. Skipping tiling for %s.", source)
                return

            # Calculate necessary size for image to be perfectly tileable with given stride and tile size
            # Number of steps (strides) needed in x and y
            # sx = (original_width - tile_w) / _stride_w  (can be float)
            # sy = (original_height - tile_h) / _stride_h (can be float)
            # Ensure we have enough pixels for the last tile:
            # new_width = _stride_w * floor(sx) + tile_w if sx is not integer, else original_width
            # A common approach is to resize so that (new_dim - tile_dim) % stride == 0

            # Simplified resizing: ensure width/height can be expressed as N*stride + tile_dim - stride
            # Or, more simply, ensure (dim - tile_dim) is a multiple of stride, if dim >= tile_dim
            # new_width = tile_w + math.ceil(max(0, original_width - tile_w) / _stride_w) * _stride_w
            # new_height = tile_h + math.ceil(max(0, original_height - tile_h) / _stride_h) * _stride_h

            # The original resizing logic:
            # sx,sy = (width-tile_w)/stride_w, (height-tile_h)/stride_h
            # resized = img.resize((stride_w*round(sx)+tile_w,stride_h*round(sy)+tile_h))
            # This logic ensures that (new_dim - tile_dim) is a multiple of stride_dim.
            # Let's stick to it but be careful with potential negative sx/sy if tile > original dim.

            if original_width < tile_w or original_height < tile_h:
                logging.warning(f"Image {source.name} ({original_width}x{original_height}) is smaller than tile size "
                                f"({tile_w}x{tile_h}). It will be padded/handled by crop. Consider adjusting tile size.")
                # If image is smaller than tile, original tiling logic might behave unexpectedly with resize.
                # A common approach for smaller images is to pad them to tile_w x tile_h.
                # For now, let original logic proceed, but this is a point of caution.
                # The original resize logic will likely expand it if sx/sy are negative.

            sx = (original_width - tile_w) / _stride_w if _stride_w > 0 else 0
            sy = (original_height - tile_h) / _stride_h if _stride_h > 0 else 0

            # Ensure sx, sy are not negative for round() if original dim < tile_dim
            # round(negative) behaves differently. We want number of steps.
            # If original_width < tile_w, sx is negative. round(sx) could be -1 or 0.
            # Number of steps should be at least 1 (one tile).

            # Corrected calculation for target dimensions for resizing
            # target_width ensures that (target_width - tile_w) is a multiple of stride_w
            # and target_width >= original_width (or at least tile_w)
            if original_width <= tile_w:
                target_resized_width = tile_w
            else:
                target_resized_width = _stride_w * round(sx) + tile_w
                if target_resized_width < original_width : # if rounding down made it too small
                    target_resized_width = _stride_w * (round(sx)+1) + tile_w

            if original_height <= tile_h:
                target_resized_height = tile_h
            else:
                target_resized_height = _stride_h * round(sy) + tile_h
                if target_resized_height < original_height:
                     target_resized_height = _stride_h * (round(sy)+1) + tile_h

            resized_img = img.resize((target_resized_width, target_resized_height))

            # Use dimensions of the (potentially) resized image for tiling calculations
            current_width, current_height = resized_img.size
            logging.info(f'Tiling {source.name}: original size ({original_width}x{original_height}), '
                         f'resized to ({current_width}x{current_height}) for tiling.')

            num_x_steps = round((current_width - tile_w) / _stride_w) + 1 if current_width >= tile_w else 1
            num_y_steps = round((current_height - tile_h) / _stride_h) + 1 if current_height >= tile_h else 1

            tiles = {}
            for x_step_idx in range(num_x_steps):
                for y_step_idx in range(num_y_steps):
                    x_coord = x_step_idx * _stride_w
                    y_coord = y_step_idx * _stride_h

                    # Define the box to cut out
                    box = (x_coord, y_coord, x_coord + tile_w, y_coord + tile_h)
                    tile = resized_img.crop(box)
                    # Use step indices for filenames for easier sorting and reconstruction
                    tiles[f'{y_step_idx}-{x_step_idx}'] = tile

            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)

            for key, tile_img in tiles.items():
                tile_img.save(dest / f"{source.stem}-{key}{source.suffix}") # Keep original suffix if not PNG

            # write metadata for entire image reconstruction
            metadata = {
                'original_filename': source.name,
                'original_width': original_width,
                'original_height': original_height,
                'resized_for_tiling_width': current_width, # Store this for precise reconstruction
                'resized_for_tiling_height': current_height,
                'tile_width': tile_w,
                'tile_height': tile_h,
                'stride_width': _stride_w,
                'stride_height': _stride_h,
                'num_x_steps': num_x_steps, # Store number of steps for easier reconstruction
                'num_y_steps': num_y_steps
            }
            with open(dest / f"{source.stem}.json", 'w') as f:
                json.dump(metadata, f, indent=4)

    except FileNotFoundError:
        logging.error(f"Source image not found: {source}")
    except Exception as e:
        logging.error(f"Error tiling image {source}: {e}")


def tile_image(source: str, dest: str, tile_w: int, tile_h: int, image_type: str,
               stride_w: Optional[int] = None, stride_h: Optional[int] = None) -> None:
    """
    Tiles a source image or all images in a source directory into smaller pieces.

    Args:
        source (str): Path to the source image file or directory of images.
        dest (str): Path to the destination directory where tiles will be saved.
        tile_w (int): Width of each tile.
        tile_h (int): Height of each tile.
        image_type (str): Image file extension (e.g., 'png', 'jpg') to process
                          if `source` is a directory.
        stride_w (Optional[int]): Stride width for tiling. Defaults to `tile_w`.
        stride_h (Optional[int]): Stride height for tiling. Defaults to `tile_h`.
    """
    src_path = Path(source)
    dest_path = Path(dest)
    
    if not dest_path.exists():
        logging.info(f"Creating destination directory: {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        if src_path.suffix.lower().strip('.') == image_type.lower():
            __tile_image(src_path, dest_path, tile_w, tile_h, stride_w, stride_h)
        else:
            logging.warning(f"Source file {src_path} does not match type '{image_type}'. Skipping.")
    elif src_path.is_dir():
        logging.info(f"Processing all *.{image_type} files in directory: {src_path}")
        for file_path in src_path.glob(f'*.{image_type}'):
            __tile_image(file_path, dest_path, tile_w, tile_h, stride_w, stride_h)
    else:
        logging.error(f"Source path {src_path} is not a valid file or directory.")


def reconstruct_image(source: str, dest: str, overlap_mode: str) -> None:
    """
    Reconstructs original image(s) from a directory of tiles and their metadata.

    It reads all .png tiles and corresponding .json metadata files from the `source`
    directory. For each set of tiles belonging to an original image, it stitches
    them together. Overlapping regions are handled based on `overlap_mode`.

    Args:
        source (str): Path to the directory containing the tiled images (.png)
                      and their metadata (.json) files.
        dest (str): Path to the directory where reconstructed images will be saved.
        overlap_mode (str): Method to handle overlapping regions between tiles.
                            Supported modes: 'max' (takes the maximum pixel value),
                            'avg' (averages pixel values).

    Raises:
        FileNotFoundError: If the source directory or essential metadata/tile files are not found.
        ValueError: If an unsupported `overlap_mode` is provided.
        Exception: For other errors during reconstruction (e.g., JSON parsing).
    """
    src_path = Path(source)
    dest_path = Path(dest)

    if not src_path.is_dir():
        raise FileNotFoundError(f"Source directory for tiles not found: {src_path}")

    if not dest_path.exists():
        logging.info(f"Creating destination directory for reconstructed images: {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)

    # Group tile files by original image stem
    # Assumes tile names are like: original_stem-rowidx-colidx.png
    tile_groups = {}
    for tile_path in src_path.glob('*-*-*.png'): # Pattern like 'stem-row-col.png'
        try:
            parts = tile_path.stem.split('-')
            row_idx = int(parts[-2])
            col_idx = int(parts[-1])
            original_stem = "-".join(parts[:-2])

            if original_stem not in tile_groups:
                tile_groups[original_stem] = []
            tile_groups[original_stem].append({'r': row_idx, 'c': col_idx, 'path': tile_path})
        except (IndexError, ValueError):
            logging.warning(f"Could not parse tile indices from filename: {tile_path.name}. Skipping.")
            continue

    if not tile_groups:
        logging.info(f"No valid tile files found in {src_path} to reconstruct.")
        return

    for original_stem, tiles_info in tile_groups.items():
        logging.info(f"Reconstructing image: {original_stem}")
        metadata_path = src_path / f"{original_stem}.json"
        if not metadata_path.exists():
            logging.error(f"Metadata file {metadata_path} not found for {original_stem}. Cannot reconstruct.")
            continue
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Use resized_for_tiling dimensions for canvas, as tiles were cut from this version
            canvas_w = metadata['resized_for_tiling_width']
            canvas_h = metadata['resized_for_tiling_height']
            tile_w = metadata['tile_width']
            tile_h = metadata['tile_height']
            stride_w = metadata['stride_width']
            stride_h = metadata['stride_height']
            # original_w = metadata['original_width'] # For potential final crop to original size
            # original_h = metadata['original_height']

            # Determine image mode and channels from the first tile
            first_tile_info = next((t for t in tiles_info if t['r']==0 and t['c']==0), tiles_info[0]) # Get 0-0 or first available
            with Image.open(first_tile_info['path']) as sample_tile:
                img_mode = sample_tile.mode
                num_channels = len(sample_tile.getbands())

            # Initialize canvas with a value indicating "no data"
            # Using float for averaging, then convert to uint8
            if num_channels == 1 or img_mode == 'L': # Grayscale
                canvas = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
                pixel_counts = np.zeros((canvas_h, canvas_w), dtype=np.float32) # For averaging
            else: # RGB or other multi-channel
                canvas = np.full((canvas_h, canvas_w, num_channels), -1.0, dtype=np.float32)
                pixel_counts = np.zeros((canvas_h, canvas_w, num_channels), dtype=np.float32)


            for tile_info in sorted(tiles_info, key=lambda t: (t['r'], t['c'])):
                r, c, tile_path = tile_info['r'], tile_info['c'], tile_info['path']
                try:
                    with Image.open(tile_path) as tile_pil:
                        tile_np = np.array(tile_pil, dtype=np.float32)

                    x_pos, y_pos = c * stride_w, r * stride_h

                    # Region in canvas to place/merge the tile
                    canvas_slice = canvas[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w]
                    tile_slice = tile_np[0:canvas_slice.shape[0], 0:canvas_slice.shape[1]] # Ensure tile slice matches canvas slice if at edge

                    # Mask of pixels in canvas_slice already filled by previous tiles
                    # -1 was our initial "no data" marker
                    filled_mask = canvas_slice != -1.0

                    # Pixels to be newly filled by the current tile (where canvas is -1)
                    newly_filled_mask = ~filled_mask

                    # Apply new data directly where canvas was empty
                    canvas_slice[newly_filled_mask] = tile_slice[newly_filled_mask]
                    pixel_counts[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w][newly_filled_mask] += 1


                    # Handle overlaps based on selected mode
                    if overlap_mode == 'max':
                        canvas_slice[filled_mask] = np.maximum(canvas_slice[filled_mask], tile_slice[filled_mask])
                        # For max, pixel_counts update is tricky if we are not averaging.
                        # Let's assume for 'max', we don't need to average, so count stays 1 unless it was 0.
                        # This part of logic might need refinement based on exact expectation of 'max' with multiple overlaps.
                    elif overlap_mode == 'avg':
                        canvas_slice[filled_mask] += tile_slice[filled_mask] # Sum overlapping values
                        pixel_counts[y_pos : y_pos + tile_h, x_pos : x_pos + tile_w][filled_mask] += 1
                    else:
                        raise ValueError(f"Unknown overlap mode: {overlap_mode}. Supported modes are 'max', 'avg'.")

                except FileNotFoundError:
                    logging.warning(f"Tile file not found during reconstruction: {tile_path}. Skipping.")
                    continue
                except Exception as e:
                    logging.error(f"Error processing tile {tile_path}: {e}")
                    continue

            # Finalize image: average where counts > 1 for 'avg' mode, handle -1s, convert to uint8
            if overlap_mode == 'avg':
                # Avoid division by zero, ensure counts are at least 1 where canvas is not -1
                valid_pixels_mask = canvas != -1.0
                # Ensure pixel_counts is not zero where valid_pixels_mask is true, to prevent division by zero
                # This can happen if a pixel was marked valid but somehow its count remained 0.
                # For safety, where canvas is valid but count is zero, set count to 1 (no averaging needed).
                if num_channels > 1 or img_mode != 'L':
                    for ch in range(num_channels):
                         channel_counts = pixel_counts[:,:,ch]
                         channel_canvas = canvas[:,:,ch]
                         channel_counts[(channel_canvas != -1.0) & (channel_counts == 0)] = 1.0
                         canvas[:,:,ch] = np.where(channel_counts > 0, channel_canvas / channel_counts, 0) # Average or keep 0 if count somehow still 0
                else: # Grayscale
                    pixel_counts[(canvas != -1.0) & (pixel_counts == 0)] = 1.0
                    canvas = np.where(pixel_counts > 0, canvas / pixel_counts, 0)


            canvas[canvas == -1.0] = 0 # Replace any remaining "no data" markers with black

            final_image_np = np.clip(canvas, 0, 255).astype(np.uint8)

            # Optional: Crop back to original_width, original_height if metadata provides it
            # and if resized_for_tiling_width/height differs from original_width/height.
            # final_image_np = final_image_np[0:metadata['original_height'], 0:metadata['original_width']]

            final_image_pil = Image.fromarray(final_image_np, mode=img_mode)
            output_file_path = dest_path / f"{original_stem}_reconstructed.png"
            final_image_pil.save(output_file_path)
            logging.info(f"Saved reconstructed image: {output_file_path}")

        except FileNotFoundError:
            logging.error(f"Metadata file {metadata_path} not found or other critical file missing for {original_stem}.")
        except Exception as e:
            logging.error(f"Failed to reconstruct {original_stem}: {e}")


if __name__=="__main__":
    ap = argparse.ArgumentParser(
        description="Tiles large images or reconstructs images from tiles."
    )
    ap.add_argument('--option', required=True, choices=['tile', 'reconstruct'],
                        help="Operation to perform: 'tile' or 'reconstruct'.")
    ap.add_argument('--src', required=True,
                        help="Source path: image file or directory for 'tile'; "
                             "directory of tiles for 'reconstruct'.")
    ap.add_argument('--dest', required=True,
                        help="Destination path: directory for output tiles/metadata for 'tile'; "
                             "directory for reconstructed images for 'reconstruct'.")
    ap.add_argument('--w', type=int, default=224, help="Tile width (default: 224, for 'tile' option).")
    ap.add_argument('--h', type=int, default=224, help="Tile height (default: 224, for 'tile' option).")
    ap.add_argument('--stride_w', type=int, default=None,
                        help="Stride width for tiling. Defaults to tile width (no overlap). (for 'tile' option)")
    ap.add_argument('--stride_h', type=int, default=None,
                        help="Stride height for tiling. Defaults to tile height (no overlap). (for 'tile' option)")
    ap.add_argument('--overlap_mode', default='max', choices=['max', 'avg'],
                        help="Mode for handling overlaps during reconstruction: 'max' or 'avg'. (default: 'max', for 'reconstruct' option)")
    ap.add_argument('--type', default='png',
                        help="Image type/extension (e.g., 'png', 'jpg') to process when tiling a directory. (default: 'png', for 'tile' option)")

    args = ap.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.option == 'tile':
        if args.w <= 0 or args.h <= 0:
            logging.error("Tile width (--w) and height (--h) must be positive.")
            exit(1)
        if args.stride_w is not None and args.stride_w <= 0:
            logging.error("Stride width (--stride_w) must be positive if specified.")
            exit(1)
        if args.stride_h is not None and args.stride_h <= 0:
            logging.error("Stride height (--stride_h) must be positive if specified.")
            exit(1)

        tile_image(args.src, args.dest, args.w, args.h, args.type, args.stride_w, args.stride_h)
    elif args.option == 'reconstruct':
        reconstruct_image(args.src, args.dest, args.overlap_mode)
    else:
        # Should not be reached due to argparse choices
        logging.error(f"Invalid option: {args.option}")
        ap.print_help()
