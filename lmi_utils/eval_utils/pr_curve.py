"""
Calculates and plots Precision-Recall (PR) curves for object detection and
segmentation models.

This script processes ground truth and model prediction data from CSV files,
calculates precision and recall values at various confidence thresholds,
and then plots these values to generate PR curves. It supports both bounding
box and polygon mask annotations.

The script utilizes helper functions from `label_utils.csv_utils` for loading
CSV data and `label_utils.shapes` for representing Rect and Mask objects.
Shared IoU calculation functions (`bbox_iou`, `polygon_iou`, `polygon_ious`)
are also included.

Input CSV files are expected to be parsable by `csv_utils.load_csv`.

Command-line arguments:
  --model_csv: Path to the CSV file containing model prediction annotations. (required)
  --label_csv: Path to the CSV file containing ground truth label annotations. (required)
  --path_out: Path to the directory where the output PR curve plots (PNG files)
              will be saved. (required)
  --skip_classes: Comma-separated list of object class names to ignore during
                  PR calculation and plotting. (default: '')
  --threshold_iou: The IoU threshold to consider a detection as a true positive.
                   (default: 0.5)
  --image_level: If specified, calculates precision and recall at the image level
                 instead of the object instance level. (default: False)

Example usage:
  python pr_curve.py --model_csv /path/to/model_predictions.csv \
                     --label_csv /path/to/ground_truth_labels.csv \
                     --path_out /path/to/output_plots \
                     --threshold_iou 0.75 \
                     --skip_classes "ignore_this_class"
"""
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
import sys

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


def precision_recall(label_dt:dict, pred_dt:dict, class_map:dict, threshold_iou:float=0.5, threshold_conf:float=0.1, skip_classes:list=[], image_level:bool=False) -> tuple:
    """
    Calculates precision and recall at specific IoU and confidence thresholds.

    This function iterates through images and classes, comparing predictions against
    ground truth labels. It determines true positives (TP), false positives (FP),
    and false negatives (FN) based on the provided IoU and confidence thresholds.
    It can operate at an object instance level or an image level.

    Args:
        label_dt (dict): Dictionary mapping filenames to lists of ground truth Shape
                         objects (Rect or Mask from `label_utils.shapes`).
        pred_dt (dict): Dictionary mapping filenames to lists of predicted Shape objects.
                        Predicted shapes must have a 'confidence' attribute.
        class_map (dict): Dictionary mapping class names to class IDs. Used to iterate
                          through all known classes.
        threshold_iou (float, optional): The IoU threshold above which a prediction
                                         is considered a true positive if matched with a
                                         ground truth object of the same class. Defaults to 0.5.
        threshold_conf (float, optional): The confidence score threshold above which a
                                          prediction is considered for evaluation.
                                          Defaults to 0.1.
        skip_classes (list, optional): A list of class names to ignore during calculation.
                                       Defaults to [].
        image_level (bool, optional): If True, calculates TP/FP/FN on an image basis
                                      (i.e., does the image contain a TP/FP/FN for a class)
                                      rather than an individual object instance basis.
                                      Defaults to False.

    Returns:
        tuple:
            - P (dict): A dictionary mapping class names (and 'all') to their
                        precision values.
            - R (dict): A dictionary mapping class names (and 'all') to their
                        recall values.
            - Err (dict): A dictionary mapping class names to their image-level error
                          rate (FN_images / total_images). This is more relevant when
                          `image_level` is True or for understanding per-class miss rates.
    """

    def mask_to_np(shapes):
        """Converts a list of Mask shapes to a list of NumPy arrays of vertices."""
        masks = []
        for shape in shapes:
            cur = np.empty((0,2), dtype=np.int32)
            if not isinstance(shape, Mask) or not hasattr(shape, 'X') or not hasattr(shape, 'Y'):
                continue
            if not (hasattr(shape.X, '__iter__') and hasattr(shape.Y, '__iter__') and len(shape.X) == len(shape.Y')):
                # print(f"[Warning] Invalid/empty coordinates in Mask object for shape: {shape}")
                continue
            for x,y in zip(shape.X,shape.Y):
                cur = np.concatenate((cur,[[x,y]]),axis=0)
            if cur.shape[0] > 0:
                 masks.append(cur)
        return masks
    
    def bboxs_to_shapes_for_iou(bbox_shapes):
        """
        Converts a list of Rect shapes to a format suitable for IoU calculation
        (either list of np.array vertices for polygons or list of [x1,y1,x2,y2] for bboxes).
        This function adapts Rect shapes to be [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for polygon_ious
        or keeps them as [x1,y1,x2,y2] for bbox_iou.
        For this PR curve context, if we go into mask mode, bboxes become polygons.
        """
        output_shapes = []
        for shape in bbox_shapes:
            if not isinstance(shape, Rect) or not hasattr(shape, 'up_left') or not hasattr(shape, 'bottom_right'):
                continue
            if None in shape.up_left or None in shape.bottom_right: # check for None coordinates
                # print(f"[Warning] Skipping Rect with None coordinates: {shape}")
                continue
            x1,y1 = shape.up_left
            x2,y2 = shape.bottom_right
            # For polygon_ious, convert to polygon vertices
            output_shapes.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32))
        return output_shapes

    def get_shape_coords_and_classes(shapes_list, is_mask_mode_pred):
        """
        Extracts coordinates (bboxes or masks) and class labels from a list of Shape objects.
        Also extracts confidences for predictions.
        In mask mode, all label shapes are converted to masks (polygons).
        """
        s_coords, s_classes, s_confs = [], [], []
        is_actually_mask_type = [] # Tracks if original type was Mask

        for s in shapes_list:
            if isinstance(s, Mask) and hasattr(s, 'X') and hasattr(s, 'Y') and s.X is not None and s.Y is not None:
                coords_np = mask_to_np([s])
                if coords_np:
                    s_coords.append(coords_np[0])
                    s_classes.append(s.category)
                    if hasattr(s, 'confidence'): s_confs.append(s.confidence)
                    is_actually_mask_type.append(True)
            elif isinstance(s, Rect) and hasattr(s, 'up_left') and hasattr(s, 'bottom_right') and \
                 s.up_left is not None and s.bottom_right is not None and None not in s.up_left and None not in s.bottom_right:
                if is_mask_mode_pred: # Convert Rect to polygon if predictions are masks
                    poly_coords = bboxs_to_shapes_for_iou([s])
                    if poly_coords:
                        s_coords.append(poly_coords[0])
                        is_actually_mask_type.append(False) # Original was Rect, now a polygon
                else: # Bbox mode
                    s_coords.append(np.array(s.up_left + s.bottom_right, dtype=np.int32))
                    is_actually_mask_type.append(False)
                s_classes.append(s.category)
                if hasattr(s, 'confidence'): s_confs.append(s.confidence)

        # If not in mask_mode_pred and no masks were found, s_coords for bboxes should be [N,4]
        # If in mask_mode_pred, all s_coords are lists of polygon vertices.
        # This needs careful handling for iou functions.
        # For simplicity, this helper will return list of coords. Conversion to np.array happens later.

        return s_coords, np.array(s_classes), np.array(s_confs), is_actually_mask_type

    TP, FP, GT, FN = collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)
    TP_im, FP_im, GT_im, FN_im = collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)
    
    fnames = set(list(label_dt.keys()) + list(pred_dt.keys()))
    total_imgs = len(fnames)

    for fname in fnames:
        current_labels = label_dt.get(fname, [])
        current_preds = pred_dt.get(fname, [])

        # Determine if prediction mode is mask based on first valid prediction shape
        is_pred_mask_mode = False
        for p_shape in current_preds:
            if isinstance(p_shape, Mask) and hasattr(p_shape, 'X') and p_shape.X is not None:
                is_pred_mask_mode = True
                break
            if isinstance(p_shape, Rect): # If Rects are first, assume bbox unless a Mask is found
                break

        coords_label, class_label, _, _ = get_shape_coords_and_classes(current_labels, is_pred_mask_mode)
        coords_pred, class_pred, conf_pred, _ = get_shape_coords_and_classes(current_preds, is_pred_mask_mode)

        # Filter predictions by confidence threshold
        if conf_pred.size > 0:
            conf_mask = conf_pred >= threshold_conf
            coords_pred = [coords_pred[i] for i, take in enumerate(conf_mask) if take]
            class_pred = class_pred[conf_mask]

        # Unique classes present in this image (labels or filtered preds)
        unique_classes = set(list(class_label) + list(class_pred))

        for c in unique_classes:
            if c in skip_classes:
                continue

            # Filter coordinates for the current class c
            c_coords_label = [coords_label[i] for i, cls in enumerate(class_label) if cls == c]
            c_coords_pred = [coords_pred[i] for i, cls in enumerate(class_pred) if cls == c]

            num_gt_for_class = len(c_coords_label)
            num_pred_for_class = len(c_coords_pred)

            GT[c] += num_gt_for_class
            if num_gt_for_class > 0:
                 GT_im[c] += 1 # Image has ground truth for this class

            if num_pred_for_class == 0: # No predictions for this class (or all filtered by conf)
                if num_gt_for_class > 0:
                    FN[c] += num_gt_for_class
                    FN_im[c] += 1 # Image has FNs for this class
                continue # No TPs or FPs if no predictions

            if num_gt_for_class == 0: # All predictions for this class are FPs
                FP[c] += num_pred_for_class
                if num_pred_for_class > 0:
                    FP_im[c] += 1 # Image has FPs for this class
                continue

            # At this point, num_gt_for_class > 0 and num_pred_for_class > 0 for class c
            # Calculate IoUs
            if is_pred_mask_mode:
                # coords are lists of polygon vertices (np.array)
                ious_matrix = polygon_ious(c_coords_pred, c_coords_label)
            else:
                # coords are bbox arrays [x1,y1,x2,y2]
                ious_matrix = bbox_iou(np.array(c_coords_pred), np.array(c_coords_label))

            if ious_matrix.size == 0: # Should not happen given checks, but as safeguard
                FP[c] += num_pred_for_class
                if num_pred_for_class > 0: FP_im[c] += 1
                FN[c] += num_gt_for_class
                if num_gt_for_class > 0: FN_im[c] += 1
                continue

            # Match predictions to ground truths
            # Each prediction can match at most one GT object (highest IoU)
            # Each GT object can be matched at most once

            # For each prediction, find its best GT match
            matched_gt_indices = np.argmax(ious_matrix, axis=1) # Index of GT with max IoU for each pred
            matched_gt_ious = np.max(ious_matrix, axis=1)    # The actual max IoU values

            # Predictions that meet IoU threshold
            tp_preds_mask = matched_gt_ious >= threshold_iou

            # Avoid double counting GTs: if multiple preds match same GT, only one is TP
            # This is a common way to resolve this:
            # Iterate through GTs. For each GT, find the best pred. If that pred's best is this GT, and IoU is good -> TP
            # Simpler approach often used: iterate predictions. If a pred is a TP, mark its matched GT as "used".

            # Current common approach (used in COCO eval for example, simplified):
            # Sort predictions by confidence (already implicitly handled by thresholding, but for AP needs full sort)
            # Iterate through predictions. If a pred matches an *unmatched* GT with IoU > thresh, it's a TP. Mark GT as matched. Else FP.
            # Unmatched GTs are FNs.

            # Simpler logic for PR at a fixed confidence threshold (as done here):
            gt_matched_flags = np.zeros(num_gt_for_class, dtype=bool)
            current_class_tp = 0

            for pred_idx in range(num_pred_for_class):
                if tp_preds_mask[pred_idx]: # This pred meets IoU threshold with *some* GT
                    gt_idx = matched_gt_indices[pred_idx]
                    if not gt_matched_flags[gt_idx]: # If this GT hasn't been matched yet
                        gt_matched_flags[gt_idx] = True
                        current_class_tp += 1
                    # else: this pred is an FP because its best GT match was already taken by a higher-conf pred (not handled here as conf is fixed)
                    # or, if not sorting by conf, it's an FP because another pred also matched this GT.
                    # For fixed confidence, if multiple preds match same GT well, they are often all TP against that one GT in simpler PR.
                    # To be more COCO-like, only one pred should claim a GT.
                    # Let's stick to: a pred is TP if it matches *any* GT well enough for now, and GTs can be multi-matched for this basic P/R.
                    # This means TP can be > num_gt_for_class if many preds match one GT. This is not standard.

            # Revision: A common way for P/R (not mAP) is:
            # TP = number of predictions that match a GT with IoU >= threshold.
            # FP = number of predictions that do NOT match any GT with IoU >= threshold.
            # FN = number of GTs that are NOT matched by any prediction with IoU >= threshold.

            tp_for_class_c = 0
            # For each GT, has it been matched by any prediction?
            gt_has_match = np.zeros(num_gt_for_class, dtype=bool)
            for gt_idx in range(num_gt_for_class):
                if ious_matrix.shape[0] > 0: # If there are predictions
                    if np.max(ious_matrix[:, gt_idx]) >= threshold_iou:
                        gt_has_match[gt_idx] = True

            fn_for_class_c = num_gt_for_class - np.sum(gt_has_match)
            FN[c] += fn_for_class_c
            if fn_for_class_c > 0: FN_im[c] +=1


            # For each Prediction, is it a TP or FP?
            # A prediction is TP if it matches a GT (that it has highest IoU with) AND that IoU is >= threshold
            # This can still lead to multiple predictions matching the same GT.
            # Standard P/R: TP = sum over GTs of (1 if matched else 0). FP = N_preds - TP.
            # This means TP <= N_GTs.
            # Let's use a matching algorithm like Hungarian, or simplify:
            # Iterate predictions. If a pred matches a GT_i with IoU > thresh, and GT_i is not yet "claimed", claim GT_i and count TP.

            # Using a simplified greedy approach for TP/FP based on predictions:
            # Each prediction is either a TP or FP.
            # TP = number of predictions that successfully match a unique GT object.
            # This requires careful handling of shared matches.

            # Alternative: TP is the number of GT objects that were successfully detected.
            tp_for_class_c = np.sum(gt_has_match) # Number of GTs that were detected
            TP[c] += tp_for_class_c
            if tp_for_class_c > 0: TP_im[c] +=1

            # FP = Total predictions for this class - Predictions that were TPs
            # This definition of TP (number of detected GTs) means a single pred can make a GT detected.
            # If multiple preds hit same GT, TP for GT is 1. What about FPs?
            # If N_preds_for_GT_X > 1, then N_preds_for_GT_X - 1 are effectively FPs for that specific GT.

            # Let's use the definition from many challenges:
            # A prediction is a TP if it has IoU > threshold with a GT *and* that GT has not been claimed by another *higher-scoring* prediction.
            # Since we are at a fixed confidence, we can simplify.
            # We need to ensure each GT is matched at most once.

            # Greedy matching of predictions to GTs
            # Sort predictions by their max IoU with any GT (descending) or by confidence (if available and not fixed)
            # For now, no secondary sort.

            # Reset gt_matched_flags for this refined TP/FP counting for predictions
            gt_claimed_for_tpfp_count = np.zeros(num_gt_for_class, dtype=bool)
            local_tp_preds = 0
            for pred_idx in range(num_pred_for_class):
                best_gt_match_for_this_pred = np.argmax(ious_matrix[pred_idx, :])
                iou_with_best_gt = ious_matrix[pred_idx, best_gt_match_for_this_pred]

                if iou_with_best_gt >= threshold_iou:
                    if not gt_claimed_for_tpfp_count[best_gt_match_for_this_pred]:
                        gt_claimed_for_tpfp_count[best_gt_match_for_this_pred] = True
                        local_tp_preds +=1
                    # else: this pred matches a GT that's already claimed by another pred for TP counting.
                    # This prediction becomes an FP unless specific duplicate detection logic is applied.
                    # For PR curve, typically it is counted as FP if its target GT is already covered.

            # TP[c] should be number of GTs matched. This is `np.sum(gt_claimed_for_tpfp_count)`
            # Or, TP is local_tp_preds if we define TP from prediction perspective.
            # Let's redefine: TP = number of GTs that are correctly matched.
            # FN = total GTs - TP.
            # FP = total Predictions - TP_preds (where TP_preds is number of preds that became TPs)

            # Let's stick to the definition where TP is number of GT instances found.
            # TP[c] += tp_for_class_c; (already done, tp_for_class_c = np.sum(gt_has_match))
            # FP calculation:
            # Number of predictions whose max IoU with any GT of that class is < threshold_iou
            fp_count_for_class_c = 0
            if num_pred_for_class > 0 and ious_matrix.shape[1]>0 : # Need GTs to calculate IoU against
                preds_max_iou_with_any_gt = np.max(ious_matrix, axis=1)
                fp_count_for_class_c = np.sum(preds_max_iou_with_any_gt < threshold_iou)

                # Add FPs also for predictions that matched a GT that was already claimed by another prediction
                # This requires a more complex assignment (e.g. using sorted predictions by score)
                # For fixed confidence, if multiple predictions hit the same GT with IoU > thresh,
                # one is TP, others are FP.
                # `local_tp_preds` (from greedy assignment) correctly gives num of preds that are TP.
                fp_count_for_class_c = num_pred_for_class - local_tp_preds
            elif num_pred_for_class > 0 and num_gt_for_class == 0 : # Preds but no GTs of this class
                fp_count_for_class_c = num_pred_for_class

            FP[c] += fp_count_for_class_c
            if fp_count_for_class_c > 0 : FP_im[c] +=1

    print(f'Threshold IoU: {threshold_iou:.2f}, Threshold Conf: {threshold_conf:.2f}')
    epsilon = 1e-16
    P, R, Err_rate = {}, {}, {} # Err_rate for image-level error (FN images / Total images)

    overall_tp, overall_fp, overall_gt_instances = 0, 0, 0

    # Use class_map keys to iterate through all known classes from labels
    # This ensures classes with no predictions are still processed for recall calculation
    for c in class_map:
        if c in skip_classes:
            continue

        if image_level:
            # Image-level: TP if class c is correctly detected in image, FP if wrongly detected, FN if missed.
            # GT_im[c] = # images that should have class c
            # TP_im[c] = # images where class c was correctly detected (at least one TP instance)
            # FP_im[c] = # images where class c was detected but no GT (or only FP instances)
            # FN_im[c] = # images where class c was GT but not detected at all
            # This interpretation needs TP_im, FP_im, FN_im to be carefully calculated based on image outcomes.
            # The current TP_im increments if any TP instance occurs. FP_im if any FP. FN_im if any FN.
            # This can lead to an image being TP_im and FP_im and FN_im for the same class if complex scene.
            # Standard image-level P/R is usually simpler: e.g. for classification, is class X present? Y/N.
            # For detection, "image contains at least one correct detection of class X".
            # Let's assume the current _im stats are about "images with at least one TP/FP/FN instance"
            # This is not standard image-level P/R. For now, will report instance-level if image_level is False.
            # If image_level is True, the definition of TP_im, FP_im, FN_im must be very clear.
            # The original code was: tp,fp,gt,fn = TP_im[c],FP_im[c],GT_im[c],FN_im[c]
            # This implies GT_im is total images with class c.
            # Let's refine image-level definition:
            # An image is a True Positive for class C if it contains at least one GT of C AND at least one Pred of C is a TP for C.
            # An image is a False Positive for class C if it has no GT of C, but has a Pred of C (that passes conf). OR if all Preds of C are FPs.
            # An image is a False Negative for class C if it has GT of C, but no Pred of C is a TP for C.

            # For now, sticking to previous logic for _im if image_level=True, but it's flawed for standard P/R.
            # The most meaningful P/R is usually instance-level unless specified.
            tp = TP_im[c] if image_level else TP[c]
            fp = FP_im[c] if image_level else FP[c]
            # GT for image level is number of images that *should* contain class c.
            gt_val = GT_im[c] if image_level else GT[c] # GT[c] is total instances. GT_im[c] is images with instances.
            fn = FN_im[c] if image_level else FN[c] # Similar issue for FN.
        else: # Instance-level
            tp = TP[c]
            fp = FP[c]
            gt_val = GT[c]
            # FN[c] is already calculated as GT instances not covered by TPs.
            # So, Recall = TP / (TP + FN) which is TP / GT_instances

        P[c] = tp / (tp + fp + epsilon)
        R[c] = tp / (gt_val + epsilon) # Recall is tp / total actual positives

        # Err_rate: fraction of images where class c was present but entirely missed (all its instances were FNs)
        # Or, if image_level, FN_im[c] (images with FNs) / GT_im[c] (images that should have c)
        Err_rate[c] = (FN_im[c] / (GT_im[c] + epsilon)) if GT_im[c] > 0 else 0.0

        print(f"Class {c}: Precision: {P[c]:.4f}, Recall: {R[c]:.4f}, Error Rate (image FN/image GT): {Err_rate[c]:.4f}")
        if not image_level: # Only sum for overall instance-level metrics
            overall_tp += tp
            overall_fp += fp
            overall_gt_instances += gt_val

    if not image_level:
        P['all'] = overall_tp / (overall_tp + overall_fp + epsilon)
        R['all'] = overall_tp / (overall_gt_instances + epsilon)
        # Err_rate for 'all' is not straightforward, typically not reported as a simple average.
        # Could be overall FN instances / overall GT instances.
        # Or average of per-class error rates. For now, let's omit Err_rate['all'].
        print(f"Overall (Instance Level): Precision: {P['all']:.4f}, Recall: {R['all']:.4f}")
    else:
        # For image-level, 'all' could be micro-average (summing TP_im, FP_im, GT_im across classes)
        # or macro-average (average of per-class P/R). Macro is more common.
        # Let's calculate Macro-average P and R for image-level 'all'.
        valid_precisions = [P[c] for c in class_map if c not in skip_classes and c in P]
        valid_recalls = [R[c] for c in class_map if c not in skip_classes and c in R]
        P['all'] = sum(valid_precisions) / (len(valid_precisions) + epsilon)
        R['all'] = sum(valid_recalls) / (len(valid_recalls) + epsilon)
        print(f"Overall (Image Level Macro Average): Precision: {P['all']:.4f}, Recall: {R['all']:.4f}")
        
    print('')
    return P, R, Err_rate


def plot_curve(px: np.ndarray, dt_y: dict, save_dir: str = 'my_curve.png',
               xlabel: str = 'Confidence', ylabel: str = 'Metric',
               y_range: list = [0, 1.05], step: float = 0.1, # Adjusted y_range slightly for better legend visibility
               threshold_iou: float = 0.5, title_suffix: str = ''):
    """
    Plots a set of curves (e.g., Precision-Confidence, Recall-Confidence) on a single graph.

    Each key in `dt_y` represents a class (or 'all'), and its corresponding value
    is a list of y-axis points to plot against `px` (x-axis points, typically confidence levels).

    Args:
        px (np.ndarray): The x-axis values (e.g., confidence thresholds).
        dt_y (dict): A dictionary where keys are class labels (str) and values are lists
                     of y-axis values (e.g., precision or recall scores).
        save_dir (str, optional): The full path (including filename) where the plot image
                                  will be saved. Defaults to 'my_curve.png'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Confidence'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Metric'.
        y_range (list, optional): A list specifying the [min, max] for the y-axis.
                                  Defaults to [0, 1.05].
        step (float, optional): The step for y-axis tick marks. Defaults to 0.1.
        threshold_iou (float, optional): The IoU threshold used for calculations,
                                         included in the 'all classes' legend. Defaults to 0.5.
        title_suffix (str, optional): Suffix to add to the plot title, typically indicating
                                      if it's image-level or instance-level. Defaults to ''.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True) # Increased figure size

    for k in dt_y:
        if k == 'all':
            ax.plot(px, dt_y['all'], linewidth=3, color='blue', label=f'All Classes (IoU Thresh={threshold_iou:.2f})')
        else:
            # Use a cycle of styles or colors if many classes, for now just plot
            ax.plot(px, dt_y[k], linewidth=1.5, linestyle='--', label=f'Class: {k}')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(*y_range)
    ax.set_yticks(np.arange(y_range[0], y_range[1] + step/2, step=step)) # Ensure upper y_range is included
    ax.grid(True, linestyle=':', alpha=0.7) # Added grid for readability

    plot_title = f'{ylabel} vs. {xlabel}{title_suffix}'
    ax.set_title(plot_title, fontsize=14)

    # Adjust legend position for better visibility, especially with more classes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    try:
        fig.savefig(save_dir, dpi=250, bbox_inches='tight') # Added bbox_inches for tight layout saving
        print(f"Plot saved to {save_dir}")
    except Exception as e:
        print(f"Error saving plot {save_dir}: {e}")
    plt.close(fig) # Close the figure to free memory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculates and plots Precision-Recall curves for object "
                                                 "detection/segmentation models from CSV data. Supports both "
                                                 "bounding boxes and polygon masks.")
    parser.add_argument('--model_csv', required=True, help='Path to the CSV file with model predictions.')
    parser.add_argument('--label_csv', required=True, help='Path to the CSV file with ground truth labels.')
    parser.add_argument('--path_out', required=True, help='Output directory for saving PR curve plots (PNG files).')
    parser.add_argument('--skip_classes', default='',
                        help='Comma-separated list of class names to skip (e.g., "class1,class2").')
    parser.add_argument('--threshold_iou', type=float, default=0.5,
                        help='IoU threshold for a detection to be considered a true positive (default: 0.5).')
    parser.add_argument('--image_level', action='store_true',
                        help='If set, calculate P/R at the image level instead of instance level.')
    args = parser.parse_args() # Use parse_args() directly

    threshold_iou_arg = args.threshold_iou
    model_csv_path = args.model_csv
    label_csv_path = args.label_csv
    output_path = args.path_out
    image_level_arg = args.image_level
    skip_classes_arg = args.skip_classes.split(',') if args.skip_classes else []

    if not os.path.isfile(model_csv_path):
        raise FileNotFoundError(f'Model predictions CSV not found: {model_csv_path}')
    if not os.path.isfile(label_csv_path):
        raise FileNotFoundError(f'Ground truth labels CSV not found: {label_csv_path}')
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    print("Loading label data...")
    label_dt, class_map = csv_utils.load_csv(label_csv_path)
    if not class_map:
        print("[Warning] No class map loaded from labels.csv. PR curves may be incomplete if classes are only in model_csv.")
        # Attempt to build class_map from all unique classes in label_dt if empty
        if label_dt:
            all_label_classes = set()
            for fname_labels in label_dt.values():
                for shape_obj in fname_labels:
                    all_label_classes.add(shape_obj.category)
            class_map = {cls_name: idx for idx, cls_name in enumerate(sorted(list(all_label_classes)))}
            print(f"Built class_map from label_dt: {class_map}")


    print("Loading prediction data...")
    # Pass class_map=None if it's empty to avoid issues in csv_utils if it expects a non-empty map for specific logic
    pred_dt, _ = csv_utils.load_csv(model_csv_path, class_map=class_map if class_map else None)

    # Ensure all classes from predictions are also in class_map if class_map was initially empty or incomplete
    if pred_dt:
        all_pred_classes = set()
        for fname_preds in pred_dt.values():
            for shape_obj in fname_preds:
                all_pred_classes.add(shape_obj.category)

        newly_added_classes = False
        if not class_map: # If class_map is still None or empty
            class_map = {} # Initialize

        for cls_name in all_pred_classes:
            if cls_name not in class_map:
                class_map[cls_name] = len(class_map) # Assign a new ID
                newly_added_classes = True
        if newly_added_classes:
             print(f"Updated class_map with classes from predictions: {class_map}")

    if not class_map:
        print("[Error] class_map is empty and could not be built. Cannot proceed.")
        sys.exit(1)

    # Confidence levels for PR curve points
    confidence_thresholds = np.linspace(0.05, 0.95, num=19, endpoint=True) # Refined confidence steps
    print(f'Using confidence levels for PR curve: {confidence_thresholds}')

    Precisions, Recalls, Error_Rates = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)

    for conf_thresh in confidence_thresholds:
        P_conf, R_conf, Err_conf = precision_recall(label_dt, pred_dt, class_map,
                                                    threshold_iou_arg, conf_thresh,
                                                    skip_classes_arg, image_level_arg)
        for c in class_map: # Iterate over all known classes to ensure list lengths match
            if c in skip_classes_arg: continue # Respect skip_classes
            Precisions[c].append(P_conf.get(c, 0.0)) # Default to 0.0 if class not in P_conf (e.g. no preds)
            Recalls[c].append(R_conf.get(c, 0.0))
            Error_Rates[c].append(Err_conf.get(c, 0.0)) # Default for error rate too

        # Handle 'all' class separately as it's an aggregate
        Precisions['all'].append(P_conf.get('all', 0.0))
        Recalls['all'].append(R_conf.get('all', 0.0))
        # Error_Rates typically doesn't have an 'all' that's a simple average, depends on definition.
        # If Err_conf contains 'all', it can be appended too. Let's assume it might not.

    plot_suffix = '_image_level' if image_level_arg else '_instance_level'
    title_suffix_main = ' (Image Level)' if image_level_arg else ' (Instance Level)'

    plot_curve(confidence_thresholds, Precisions,
               save_dir=os.path.join(output_path, f'precision_vs_confidence{plot_suffix}.png'),
               ylabel='Precision', xlabel='Confidence Threshold',
               threshold_iou=threshold_iou_arg, title_suffix=title_suffix_main, step=0.1)

    plot_curve(confidence_thresholds, Recalls,
               save_dir=os.path.join(output_path, f'recall_vs_confidence{plot_suffix}.png'),
               ylabel='Recall', xlabel='Confidence Threshold',
               threshold_iou=threshold_iou_arg, title_suffix=title_suffix_main, step=0.1)

    # Optional: Plot error rate if meaningful (usually for image_level or specific class insights)
    # plot_curve(confidence_thresholds, Error_Rates,
    #            save_dir=os.path.join(output_path, f'error_rate_vs_confidence{plot_suffix}.png'),
    #            ylabel='Error Rate (FN Images / GT Images)', xlabel='Confidence Threshold',
    #            threshold_iou=threshold_iou_arg, title_suffix=title_suffix_main, y_range=[0,1.05], step=0.1)

    print(f'Precision-Recall curve plots saved in {output_path}')
