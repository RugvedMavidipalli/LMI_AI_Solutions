"""
Calculates Intersection over Union (IoU) for object detection and segmentation tasks,
supporting both bounding boxes (rectangles) and polygon masks.

This script reads object annotations from two CSV files: one for ground truth labels
and one for model predictions. It compares annotations for common images and classes,
calculates IoU, and saves the results to a new CSV file. It can also generate
and save images with overlaid ground truth and prediction shapes.

The script utilizes helper functions from `label_utils.csv_utils` for loading CSV data
and `label_utils.shapes` for representing Rect and Mask objects.

Input CSV files are expected to be parsable by `csv_utils.load_csv`.

Command-line arguments:
  --path_imgs: Path to the directory containing the images. (required)
  --model_csv: Path to the CSV file containing model prediction annotations. (required)
  --label_csv: Path to the CSV file containing ground truth label annotations. (required)
  --path_out: Path to the directory where the output IoU CSV file and annotated
              images will be saved. (required)
  --skip_classes: Comma-separated list of object class names to ignore during IoU
                  calculation and plotting. (default: '')

Example usage:
  python iou_from_csv2.py --path_imgs /path/to/images \
                          --model_csv /path/to/model_predictions.csv \
                          --label_csv /path/to/ground_truth_labels.csv \
                          --path_out /path/to/output_results \
                          --skip_classes "background,ignore_this"
"""
from gc import collect
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
import sys
import csv
import cv2

#LMI packages
from label_utils import csv_utils
from label_utils.shapes import Rect, Mask


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """
    Calculates the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        bbox1 (np.ndarray): An array of shape [N, 4] representing N bounding boxes.
                           Each row is [xmin, ymin, xmax, ymax].
        bbox2 (np.ndarray): An array of shape [M, 4] representing M bounding boxes.
                           Each row is [xmin, ymin, xmax, ymax].

    Returns:
        np.ndarray: An array of shape [N, M] containing the IoU scores between
                    each pair of bounding boxes from bbox1 and bbox2.
    """
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
    
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    # Add a small epsilon to the union to avoid division by zero if union is 0
    return intersect / (union + 1e-7)


def polygon_iou(polygon_1: list, polygon_2: list) -> float:
    """
    Calculates the Intersection over Union (IoU) between two polygons.

    Args:
        polygon_1 (list): A list of [x, y] coordinates representing the vertices
                          of the first polygon. E.g., [[row1, col1], [row2, col2], ...].
        polygon_2 (list): A list of [x, y] coordinates representing the vertices
                          of the second polygon.

    Returns:
        float: The IoU score between the two polygons. Returns 0 if an error occurs
               during polygon creation (e.g., less than 3 points).
    """
    try:
        poly_1 = Polygon(polygon_1)
        poly_2 = Polygon(polygon_2)
    except Exception as e:
        #usually less than 3 points for creating the polygons
        #print(e)
        return 0

    if not poly_1.is_valid:
        poly_1 = make_valid(poly_1)
    if not poly_2.is_valid:
        poly_2 = make_valid(poly_2)

    intersection_area = poly_1.intersection(poly_2).area
    union_area = poly_1.union(poly_2).area

    if union_area == 0:
        return 0.0 # Avoid division by zero if union is zero

    iou = intersection_area / union_area
    return iou


def polygon_ious(polygons_1: list, polygons_2: list) -> np.ndarray:
    N,M = len(polygons_1), len(polygons_2)
    ious = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ious[i][j] = polygon_iou(polygons_1[i],polygons_2[j])
    return ious


def plot_shapes(image: np.ndarray, shape_label: list, class_label: list,
                shape_pred: list, class_pred: list, skip_classes: list = [],
                is_mask: bool = False):
    """
    Draws bounding boxes or polygon masks for ground truth and predictions on an image.

    Args:
        image (np.ndarray): The image on which to draw the shapes.
        shape_label (list): List of ground truth shapes (bounding boxes or masks).
                            For bboxes: [[x1,y1,x2,y2], ...]. For masks: [np.array([[x,y],...]), ...].
        class_label (list): List of class labels corresponding to ground truth shapes.
        shape_pred (list): List of predicted shapes.
        class_pred (list): List of class labels corresponding to predicted shapes.
        skip_classes (list, optional): List of class labels to ignore (not plot). Defaults to [].
        is_mask (bool, optional): If True, treats shapes as polygon masks. Otherwise, as bounding boxes.
                                 Defaults to False.
    """
    # BGR
    BLUE = (255,0,0) # Ground truth
    RED = (0,0,255)  # Predictions
    
    def plot_bboxs(image, bboxs, labels, color:tuple, pos='uleft'):
        for i in range(len(bboxs)):
            if not isinstance(bboxs[i], (list, np.ndarray)) or len(bboxs[i]) != 4:
                print(f"[Warning] Skipping invalid bbox: {bboxs[i]}")
                continue
            x1,y1,x2,y2 = map(int, bboxs[i]) # Ensure coordinates are integers for OpenCV
            label = labels[i]
            if label in skip_classes:
                continue
            uleft,bright = (x1,y1),(x2,y2)
            cv2.rectangle(image,uleft,bright,color,1)
            if pos=='uleft':
                cv2.putText(image, label, uleft, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                # Adjust position for bottom-right to ensure visibility
                text_origin = (bright[0] - (len(label) * 5), bright[1]) # Basic adjustment
                cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
    def plot_masks(image, masks, labels, color:tuple, pos='uleft'):
        for i in range(len(masks)):
            if not isinstance(masks[i], np.ndarray) or masks[i].ndim != 2 or masks[i].shape[1] != 2:
                print(f"[Warning] Skipping invalid mask: {masks[i]}")
                continue
            pts = masks[i].astype(np.int32) # Ensure coordinates are int32 for OpenCV
            label = labels[i]
            if label in skip_classes:
                continue
            uleft = tuple(pts.min(axis=0))
            bright = tuple(pts.max(axis=0))
            pts_reshaped = pts.reshape((-1, 1, 2))
            cv2.polylines(image,[pts_reshaped],True,color,1)
            if pos=='uleft':
                cv2.putText(image, label, uleft, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                text_origin = (bright[0] - (len(label) * 5), bright[1]) # Basic adjustment
                cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    if not is_mask:
        plot_bboxs(image,shape_label,class_label,color=BLUE)
        plot_bboxs(image,shape_pred,class_pred,color=RED,pos='bright')
    else:
        plot_masks(image,shape_label,class_label,color=BLUE)
        plot_masks(image,shape_pred,class_pred,color=RED,pos='bright')


def get_ious(path_imgs:str, path_out:str, label_dt:dict, pred_dt:dict, skip_classes:list=[]) -> tuple:
    """
    Calculates IoUs for each image and class between ground truth and predictions.

    It handles both bounding boxes and polygon masks. If masks are present in predictions,
    labels are also treated as masks (converting bboxes to masks if necessary).
    Annotated images are saved to `path_out`.

    Args:
        path_imgs (str): Path to the directory containing images.
        path_out (str): Path to the directory to save annotated images.
        label_dt (dict): A dictionary mapping filenames to lists of ground truth Shape objects
                         (Rect or Mask) from `csv_utils`.
        pred_dt (dict): A dictionary mapping filenames to lists of predicted Shape objects.
        skip_classes (list, optional): List of class labels to ignore. Defaults to [].

    Returns:
        tuple:
            - all_ious (dict): A dictionary mapping filenames to another dictionary,
              which maps class names to a list of IoU values for that class in the image.
              NaN values in the list indicate false negatives (ground truth objects
              not matched by any prediction).
            - all_not_nan_ious (collections.defaultdict(list)): A dictionary mapping
              class names to a list of mean IoU values (for matched objects) per image.
    """

    def mask_to_np(shapes):
        """Converts a list of Mask shapes to a list of NumPy arrays of vertices."""
        masks = []
        for shape in shapes:
            cur = np.empty((0,2), dtype=np.int32) # Ensure dtype for OpenCV
            if not isinstance(shape, Mask):
                continue
            # Ensure X and Y are iterable and of the same length
            if not (hasattr(shape.X, '__iter__') and hasattr(shape.Y, '__iter__') and len(shape.X) == len(shape.Y')):
                print(f"[Warning] Invalid/empty coordinates in Mask object for shape: {shape}")
                continue
            for x,y in zip(shape.X,shape.Y):
                cur = np.concatenate((cur,[[x,y]]),axis=0)
            if cur.shape[0] > 0: # Only add if there are points
                 masks.append(cur)
        return masks
    
    def bboxs_to_np(bbox_shapes):
        """Converts a list of Rect shapes to a list of NumPy arrays representing polygon vertices."""
        masks = []
        for shape in bbox_shapes:
            if not isinstance(shape, Rect):
                continue
            # Ensure coordinates are available
            if not (hasattr(shape, 'up_left') and hasattr(shape, 'bottom_right') and
                    len(shape.up_left) == 2 and len(shape.bottom_right) == 2) :
                print(f"[Warning] Invalid/empty coordinates in Rect object: {shape}")
                continue
            x1,y1 = shape.up_left
            x2,y2 = shape.bottom_right
            masks.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32))
        return masks
    
    all_ious = {}
    all_not_nan_ious = collections.defaultdict(list)
    # Ensure fnames considers keys from both dictionaries, handling potential missing keys
    fnames = set(list(label_dt.keys()) + list(pred_dt.keys()))

    for fname in fnames:
        is_mask = False # Default to bbox
        I = cv2.imread(os.path.join(path_imgs,fname))
        if I is None:
            print(f'[Warning] Cannot read image: {fname}, skipping.')
            continue

        # Get labels and predictions for the current file, handle missing keys
        current_labels = label_dt.get(fname, [])
        current_preds = pred_dt.get(fname, [])

        # Prepare label shapes
        bbox_label_shapes = [s for s in current_labels if isinstance(s, Rect)]
        mask_label_shapes = [s for s in current_labels if isinstance(s, Mask)]
        bbox_label_coords = np.array([shape.up_left + shape.bottom_right for shape in bbox_label_shapes], dtype=np.int32) if bbox_label_shapes else np.empty((0,4), dtype=np.int32)
        class_label_bbox = np.array([shape.category for shape in bbox_label_shapes]) if bbox_label_shapes else np.empty((0,))

        mask_label_coords_from_mask = mask_to_np(mask_label_shapes)
        class_label_mask = np.array([shape.category for shape in mask_label_shapes]) if mask_label_shapes else np.empty((0,))

        # Prepare prediction shapes
        bbox_pred_shapes = [s for s in current_preds if isinstance(s, Rect)]
        mask_pred_shapes = [s for s in current_preds if isinstance(s, Mask)]
        bbox_pred_coords = np.array([shape.up_left + shape.bottom_right for shape in bbox_pred_shapes], dtype=np.int32) if bbox_pred_shapes else np.empty((0,4), dtype=np.int32)
        class_pred_bbox = np.array([shape.category for shape in bbox_pred_shapes]) if bbox_pred_shapes else np.empty((0,))

        mask_pred_coords = mask_to_np(mask_pred_shapes)
        class_pred_mask = np.array([shape.category for shape in mask_pred_shapes]) if mask_pred_shapes else np.empty((0,))

        # Determine if we are in mask mode (if any mask predictions exist)
        if len(mask_pred_coords) > 0:
            is_mask = True
            # Convert all labels to masks if in mask mode
            # Combine original masks with bboxes converted to masks
            all_label_masks = mask_label_coords_from_mask + bboxs_to_np(bbox_label_shapes)
            all_label_classes = np.concatenate((class_label_mask, class_label_bbox)) if class_label_mask.size > 0 or class_label_bbox.size > 0 else np.empty((0,))

            # Predictions are already masks
            all_pred_masks = mask_pred_coords
            all_pred_classes = class_pred_mask
        else:
            # Bbox mode: use bbox coordinates for both labels and predictions
            # Labels: combine mask_labels (converted to bboxes - simplified here, assuming bbox is primary if no pred masks) and bbox_labels
            # For simplicity, if masks are in labels but not preds, we might only evaluate bboxes or need a conversion strategy.
            # Current logic prioritizes bbox if no prediction masks.
            # This part might need refinement if mixed label types (mask+bbox) should be handled differently in bbox mode.
            # Assuming label_dt provides consistent types or csv_utils handles conversion to common type if possible.
            # For now, using only bbox_label_coords and class_label_bbox for labels in bbox mode.
            all_label_masks = bbox_label_coords # These are actually bboxes
            all_label_classes = class_label_bbox
            all_pred_masks = bbox_pred_coords # These are actually bboxes
            all_pred_classes = class_pred_bbox

        # Plot shapes
        plot_shapes(I, all_label_masks, all_label_classes, all_pred_masks, all_pred_classes, is_mask=is_mask, skip_classes=skip_classes)
        
        outname = os.path.splitext(fname)[0]+'_iou.png'
        cv2.imwrite(os.path.join(path_out,outname),I)

        # Calculate IoU per class
        class_to_iou = {}
        # Consider all unique classes present in either labels or predictions
        unique_classes_in_file = set(list(all_label_classes) + list(all_pred_classes))

        for c in unique_classes_in_file:
            if c in skip_classes:
                continue

            # Filter shapes for the current class
            current_class_labels = [s for i, s in enumerate(all_label_masks) if all_label_classes[i] == c]
            current_class_preds = [s for i, s in enumerate(all_pred_masks) if all_pred_classes[i] == c]

            num_labels = len(current_class_labels)
            num_preds = len(current_class_preds)

            if num_labels == 0 and num_preds == 0:
                continue # No instances of this class in this image
            elif num_labels == 0: # All predictions are false positives for this class
                final_ious_nan = np.zeros(num_preds) # IoU is 0 for FPs
            elif num_preds == 0: # All labels are false negatives for this class
                final_ious_nan = np.full(num_labels, np.nan) # NaN for FNs
            else:
                if is_mask:
                    # Ensure polygons are lists of coordinates for polygon_ious
                    ious_matrix = polygon_ious(current_class_preds, current_class_labels)
                else:
                    # Ensure bboxes are np.arrays for bbox_iou
                    ious_matrix = bbox_iou(np.array(current_class_preds), np.array(current_class_labels))

                if ious_matrix.size == 0: # Should not happen if num_preds > 0
                    final_ious_nan = np.zeros(num_preds) if num_labels == 0 else np.full(num_labels, np.nan)
                else:
                    # Max IoU for each prediction (matching it to the best GT)
                    matched_ious = np.max(ious_matrix, axis=1)
                    # Calculate mean IoU only for predictions that had a match (IoU > 0)
                    # This contributes to the per-class summary statistics later.
                    valid_matched_ious = matched_ious[matched_ious > 0]
                    if valid_matched_ious.size > 0:
                        all_not_nan_ious[c].append(np.mean(valid_matched_ious))


                    final_ious_nan = matched_ious.copy()

                    # Account for false negatives: GT objects not matched by any prediction
                    # If N_gt > N_pred_matched_to_gt_for_this_class
                    # This logic is complex with one-to-many or many-to-many.
                    # Simplification: count GTs that have no pred with IoU > threshold (e.g. 0).
                    # The original code's NaN appending for N-M is a common way to represent this.
                    # It assumes each prediction can match at most one GT for this calculation.
                    # And each GT can be matched by at most one prediction for this particular list.

                    # A more robust way to handle FNs for IoU list:
                    # For each GT, find max IoU with any pred. If 0 (or below thresh), it's an FN.
                    # However, the current structure `final_ious_nan` is prediction-centric.
                    # Let's stick to the original intent for `final_ious_nan` being based on predictions,
                    # and FNs are implicitly those GTs not covered by any prediction's IoU.
                    # The NaN appending in original code was for (N_gt - N_preds)
                    if num_labels > num_preds:
                         final_ious_nan = np.concatenate((final_ious_nan, np.full(num_labels - num_preds, np.nan)))

            class_to_iou[c] = final_ious_nan
            
        all_ious[fname] = class_to_iou
    return all_ious,all_not_nan_ious


def write_to_csv(all_ious:dict, mean_ious:dict, filename:str):
    """
    Writes the calculated IoU results to a CSV file.

    The CSV includes per-image, per-class IoU values. 'fn' indicates a false negative
    (a ground truth object not detected), 'fp' indicates a false positive (a detection
    with 0 IoU to any ground truth of that class). Numerical values are the IoUs.
    Mean IoUs per class and an overall mean IoU are also written at the end.

    Args:
        all_ious (dict): A dictionary mapping filenames to class-to-IoU-list maps,
                         as returned by `get_ious`.
        mean_ious (dict): A dictionary mapping class names (and 'all') to their
                          mean IoU values.
        filename (str): The path to the output CSV file.
    """
    with open(filename, 'w', newline='') as f: # Added newline='' for csv writer
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Image Filename', 'Class', 'IoU Values (or fn/fp)']) # Header

        for im_name in all_ious:
            for category in all_ious[im_name]:
                iou_values = all_ious[im_name][category]
                # Ensure iou_values is iterable, even if it's a single numpy array from class_to_iou
                if isinstance(iou_values, np.ndarray):
                    l = iou_values.tolist()
                elif isinstance(iou_values, (list, tuple)):
                    l = list(iou_values)
                else: # Should not happen based on get_ious logic
                    l = []
                    
                # Convert IoUs to string representations: 'fn', 'fp', or float value
                l2 = ['fn' if np.isnan(x) else ('fp' if x == 0 else f"{x:.4f}") for x in l]
                writer.writerow([im_name, category] + l2)

        writer.writerow([]) # Empty line for separation
        writer.writerow(['Mean IoUs per Class:'])
        for c in mean_ious:
            if c != 'all': # Keep 'all' for the final summary
                writer.writerow([f'Mean IoU of {c}', f"{mean_ious[c]:.4f}"])

        if 'all' in mean_ious:
             writer.writerow(["Mean IoU (all classes, matched objects)", f"{mean_ious['all']:.4f}"])


if __name__ == '__main__':
    import argparse
    # Updated description to be more informative
    parse = argparse.ArgumentParser(description='Calculate Intersection over Union (IoU) for object detection/segmentation '
                                                'from model predictions and ground truth CSV files. Supports both bounding '
                                                'boxes and polygon masks. Generates an output CSV with IoU scores and '
                                                'annotated images.')
    parse.add_argument('--path_imgs', required=True, help='The path to the directory containing the images.')
    parse.add_argument('--model_csv', required=True, help='The path to the CSV file with model predictions.')
    parse.add_argument('--label_csv', required=True, help='The path to the CSV file with ground truth labels.')
    parse.add_argument('--path_out', required=True, help='The output directory for IoU CSV and annotated images.')
    parse.add_argument('--skip_classes', default='', help='Comma-separated list of classes to skip in IoU calculation (e.g., "class1,class2").')
    args = vars(parse.parse_args())

    path_imgs = args['path_imgs']
    model_csv_path = args['model_csv'] # Renamed for clarity
    label_csv_path = args['label_csv'] # Renamed for clarity
    path_out = args['path_out']
    skip_classes_str = args['skip_classes'] # Renamed for clarity

    if skip_classes_str == '':
        skip_classes = []
    else:
        skip_classes = skip_classes_str.split(',')

    if not os.path.isfile(model_csv_path):
        # Corrected variable name in error message
        raise FileNotFoundError(f'Model predictions CSV not found: {model_csv_path}')

    if not os.path.isfile(label_csv_path):
        # Corrected variable name in error message
        raise FileNotFoundError(f'Ground truth labels CSV not found: {label_csv_path}')
    
    if not os.path.isdir(path_out):
        os.makedirs(path_out, exist_ok=True) # Added exist_ok=True

    # Load data using csv_utils
    # It's assumed csv_utils.load_csv returns:
    # 1. A dict mapping filename to a list of Shape objects (Rect or Mask)
    # 2. A dict mapping class name to class ID (class_map)
    label_dt, class_map = csv_utils.load_csv(label_csv_path)
    # If class_map is None or empty from label_csv, it might need to be built or handled
    if not class_map:
        print("[Warning] class_map from label_csv is empty. Attempting to build from labels or use predictions' classes.")
        # Potentially build class_map here if necessary, or rely on pred_dt's implicit classes.
        # For now, assume pred_dt can work with an empty or specific class_map from label_dt.

    print(f'Found class map from labels: {class_map}')
    print(f'Skipping classes: {skip_classes}')

    # Load predictions, potentially using class_map from labels for consistency
    pred_dt, _ = csv_utils.load_csv(model_csv_path, class_map=class_map if class_map else None)

    all_ious, all_not_nan_ious = get_ious(path_imgs, path_out, label_dt, pred_dt, skip_classes)
    
    # Calculate mean IoU
    mean_ious = {}
    total_iou_sum = 0
    # total_iou_count counts the number of per-image-per-class mean IoUs that were calculated
    # all_not_nan_ious[c] is a list of mean IoUs for class c, one mean per image where class c was present and matched
    total_matched_iou_values = 0
    num_mean_iou_scores_recorded = 0

    for c in all_not_nan_ious:
        if all_not_nan_ious[c]:
            class_mean_iou = sum(all_not_nan_ious[c]) / len(all_not_nan_ious[c])
            mean_ious[c] = class_mean_iou
            total_matched_iou_values += sum(all_not_nan_ious[c])
            num_mean_iou_scores_recorded += len(all_not_nan_ious[c])
        else:
            mean_ious[c] = 0.0 # Class c had no matched predictions in any image

    if num_mean_iou_scores_recorded > 0:
        mean_ious['all'] = total_matched_iou_values / num_mean_iou_scores_recorded
    else:
        mean_ious['all'] = 0.0
        
    write_to_csv(all_ious, mean_ious, os.path.join(path_out,'ious.csv'))
    print(f"IoU calculation complete. Results saved to {os.path.join(path_out,'ious.csv')}")