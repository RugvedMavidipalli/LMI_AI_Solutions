"""
Calculates Intersection over Union (IoU) for object detection or segmentation tasks
based on ground truth and model prediction data provided in CSV files.

This script reads object annotations (bounding boxes or polygons) from two CSV files:
one for manually labeled ground truth and one for model predictions. It then compares
the annotations for common images and object classes, calculates the IoU for each,
and saves the results to a new CSV file. Optionally, it can render images with
overlaid annotations and save these annotated images.

The input CSV files are expected to have a specific format where each row represents
an annotation and includes:
- Image filename (ending in .png or .jpg)
- Object class name
- Shape type ('rect', 'polygon', or 'point')
- Coordinates for the shape:
    - For 'rect': 'upper left;x;y' and 'lower right;x;y'
    - For 'polygon': 'x values;x1;x2;...' and 'y values;y1;y2;...'
    - For 'point': 'cx;x' and 'cy;y'
Cells within a row are separated by semicolons.

Command-line arguments:
  --manual_csv: Path to the CSV file containing manual (ground truth) labels. (required)
  --model_csv: Path to the CSV file containing model prediction labels. (required)
  --data_dir: Path to the directory containing the images referenced in the CSV files. (required)
  --labels: Comma-separated list of object class labels to consider. If None, all unique
            labels from both CSV files will be used. (default: None)
  --output_dir: Path to the directory where the output IoU CSV file and annotated images
                will be saved. (required)
  --render: If specified, displays images with annotations during processing. (default: False)

Example usage:
  python iou_from_csv.py --manual_csv /path/to/manual_labels.csv \
                         --model_csv /path/to/model_predictions.csv \
                         --data_dir /path/to/images \
                         --output_dir /path/to/output_results \
                         --labels "car,pedestrian" \
                         --render
"""
#%%
import re
import numpy as np
import cv2
import os
import argparse
import csv


def csv_to_dictionary(csv_file: str,object_classes: list):
    """
    Converts a CSV file with image paths, labels, and ROIs to a list of dictionaries.

    Each dictionary in the list represents an object annotation and contains keys such as
    'image_file', 'obj_class', 'shape', and shape-specific coordinates
    (e.g., 'upper_left', 'lower_right' for rectangles; 'x_values', 'y_values' for polygons).

    Args:
        csv_file (str): Path to the input CSV file.
        object_classes (list): A list of object class names to look for in the CSV.
                               Annotations not matching these classes will be skipped.

    Returns:
        list: A list of dictionaries, where each dictionary represents an object annotation.

    Raises:
        Exception: If the CSV format is incorrect (e.g., image file not found,
                   unsupported shape, or badly defined object).
    """
    
    supported_shapes=['rect','polygon','point']
    list_of_dics=[]
    rows=open(csv_file).read().strip().split('\n')
    ul_found=False 
    lr_found=False 
    x_found=False 
    y_found=False
    cx_found=False
    cy_found=False

    # Step through each row and create a new dictionary when the row includes a target object
    for row in rows:
        print(row)
        if row[-1]==';':
            row=row[0:-1]
        row=row.split(';')

         # search the cells for target object classes
        obj=list(set(object_classes) & set(row))
        if not obj or len(obj)>1:
            # raise Exception(f'csv format error. Bad object definition: {obj}')
            continue
        else:
            this_obj=obj[0]

                
        # search cells for image file
        image_file=[x for x in row if (re.search('.png',x) or re.search('.jpg',x))] 
        if len(image_file) != 1:
            raise Exception('csv format error. Image file not present.')
        else:
            this_file=image_file[0]

        # search for supported shapes
        shape=list(set(supported_shapes) & set(row))
        if not shape or len(shape)>1:
            raise Exception('csv format error. Unsupported shape.')
        else:
            this_shape=shape[0]
        
        if this_shape=='rect':
            uli=[i for i,s in enumerate(row) if 'upper left' in s]
            if len(uli)==1:
                x_ul=int(row[uli[0]+1])
                y_ul=int(row[uli[0]+2])
                ul=(x_ul,y_ul)
                ul_found=True
            lri=[i for i,s in enumerate(row) if 'lower right' in s]
            if len(lri)==1:
                x_lr=int(row[lri[0]+1])
                y_lr=int(row[lri[0]+2])
                lr=(x_lr,y_lr)
                lr_found=True
            if ul_found and lr_found:
                list_of_dics.append({"image_file":this_file,"obj_class":this_obj,"shape":this_shape,"upper_left":ul,"lower_right":lr})
                ul_found=False 
                lr_found=False 

        if this_shape=='polygon':
            xind=[i for i,s in enumerate(row) if 'x values' in s]
            if len(xind)==1:          
                this_x=np.array(row[xind[0]+1:],dtype=np.uint32)
                x_found=True
            yind=[i for i,s in enumerate(row) if 'y values' in s] 
            if len(yind)==1:
                this_y=np.array(row[yind[0]+1:],dtype=np.uint32)
                y_found=True
            if x_found and y_found:
                list_of_dics.append({"image_file":this_file,"obj_class":this_obj,"shape":this_shape,"x_values":this_x,"y_values":this_y})
                x_found=False 
                y_found=False
        
        if this_shape=='point':
            cxi=[i for i,s in enumerate(row) if 'cx' in s]
            if len(cxi)==1:
                cx=int(row[cxi[0]+1])
                cx_found=True
            cyi=[i for i,s in enumerate(row) if 'cy' in s]
            if len(cyi)==1:
                cy=int(row[cyi[0]+1])
                cy_found=True
            if cx_found and cy_found:
                list_of_dics.append({"image_file":this_file,"obj_class":this_obj,"shape":this_shape,"cx":cx,"cy":cy})
                cx_found=False 
                cy_found=False 
        
    return list_of_dics

def find_class_index(target_class: str, list_of_dicts: list):
    """
    Finds all indices of dictionaries in a list that match a target object class.

    Args:
        target_class (str): The object class to search for.
        list_of_dicts (list): A list of dictionaries, where each dictionary is expected
                              to have an 'obj_class' key.

    Returns:
        list: A list of integer indices corresponding to the dictionaries that match
              the target_class. Returns an empty list if no matches are found.
    """
    out=[]
    for i,lod in enumerate(list_of_dicts):
        if lod['obj_class']==target_class:
            out.append(i)
    return out
        
def main(model_path:str,manual_path:str,data_dir:str,labels:str,output_dir:str,render:bool):
    """
    Computes IoU from CSV label files, annotates object bounding boxes/segments,
    and saves IoU results to a CSV file.

    Args:
        model_path (str): Path to the CSV file containing model predictions.
        manual_path (str): Path to the CSV file containing manual (ground truth) labels.
        data_dir (str): Path to the directory containing the images.
        labels (str or None): A comma-separated string of object classes to evaluate.
                              If None, all unique classes from both CSVs are used.
        output_dir (str): Path to the directory where the output IoU CSV and annotated
                          images will be saved.
        render (bool): If True, displays annotated images during processing.
    """
    
    if labels is None:
        object_classes_model=np.genfromtxt(model_path,delimiter=';',dtype=str)[:,1]
        object_classes_manual=np.genfromtxt(manual_path,delimiter=';',dtype=str)[:,1]
        obj_classes=np.hstack((object_classes_model,object_classes_manual))
        obj_classes=list(np.unique(obj_classes))
    else:
        try:
            obj_classes=labels.split(",")
        except:
            print(f'Incorrect labels definition: {labels}')

    manual_data=csv_to_dictionary(manual_path,obj_classes)
    model_data=csv_to_dictionary(model_path,obj_classes)
    #%% get filenames
    model_file_set=set([item['image_file'] for item in model_data]) 
    manual_file_set=set([item['image_file'] for item in manual_data])
    file_intersection=list(model_file_set & manual_file_set)
    #%% search by mask ID
    iou=[]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outfile=os.path.join(output_dir,'iou.csv')
    #%% Generate IOU results
    with open(outfile,'w',newline='') as csvfile:
        rowWriter=csv.writer(csvfile,delimiter=',')
        rowWriter.writerow(['model label',model_path])
        rowWriter.writerow(['manual label',manual_path])
        #  
        for file_x in file_intersection:
            current_model=[item for item in model_data if item['image_file']==file_x]
            current_manual=[item for item in manual_data if item['image_file']==file_x]
            image=cv2.imread(os.path.join(data_dir,current_model[0]['image_file']))
            for obj_class in obj_classes:
                i_manual=find_class_index(obj_class,current_manual)
                i_model=find_class_index(obj_class,current_model)
                if (not i_manual) and (not i_model):
                    continue
                else:
                    label = f'{obj_class}'
                    found_polygons=False
                    found_boxes=False
                    manual_mask=np.zeros(image.shape[0:2],dtype=np.uint8)
                    for i in i_manual:
                        if current_manual[i]['shape']=='polygon':
                            found_polygons=True
                            x_manual=current_manual[i]['x_values']
                            y_manual=current_manual[i]['y_values']
                            pts=np.stack((x_manual,y_manual),axis=1)
                            label_coord=pts.min(axis=0)
                            pts_manual=np.expand_dims(pts,axis=0).astype(np.int32)
                            cv2.polylines(image,pts_manual,True,(255,0,0),1)
                            cv2.fillPoly(manual_mask,pts_manual,255)
                            cv2.putText(image, label, label_coord,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
                        elif current_manual[i]['shape']=='rect':
                            found_boxes=True
                            uleft=current_manual[i]['upper_left']
                            bright=current_manual[i]['lower_right']
                            label_coord=uleft
                            cv2.rectangle(image,uleft,bright,(255,0,0),1)
                            cv2.rectangle(manual_mask,uleft,bright,255,-1)
                            cv2.putText(image, label, label_coord,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
                        else:
                            raise Exception('Unknown mask shape.')

                    model_mask=np.zeros(image.shape[0:2],dtype=np.uint8)
                    for i in i_model:
                        if found_polygons and current_model[i]['shape']=='polygon':
                            x_model=current_model[i]['x_values']
                            y_model=current_model[i]['y_values']
                            pts=np.stack((x_model,y_model),axis=1)
                            label_coord=pts.max(axis=0)
                            pts_model=np.expand_dims(pts,axis=0).astype(np.int32)
                            cv2.polylines(image,pts_model,True,(0,0,255),1)
                            cv2.fillPoly(model_mask,pts_model,255)
                            label_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.3,1)
                            label_size=label_size[0][0]
                            print(f'[INFO] Label size: {label_size}')
                            overrun=label_coord[0]+label_size
                            if overrun > image.shape[1]:
                                delta=overrun-image.shape[1]
                                image=cv2.copyMakeBorder(image,0,0,0,delta,cv2.BORDER_CONSTANT,None,(0,0,0))
                            cv2.putText(image, label, label_coord,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
                        elif found_boxes and current_model[i]['shape']=='rect':
                            uleft=current_model[i]['upper_left']
                            bright=current_model[i]['lower_right']
                            label_coord=bright
                            cv2.rectangle(image,uleft,bright,(0,0,255),1)
                            cv2.rectangle(model_mask,uleft,bright,255,-1)
                            cv2.putText(image, label, label_coord,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
                        else:
                            print(f"[INFO] skipping model, {current_model[i]['shape']} not present in labeled data.")
                    
                    image_rsz=image.copy()
                    if render:
                        cv2.imshow('Validation Window',image_rsz)
                        cv2.waitKey(100)
                    img_file=os.path.splitext(file_x)[0]+'_iou.png'
                    cv2.imwrite(os.path.join(output_dir,img_file),image_rsz)
                    
                    #compute IOU
                    union=np.sum(cv2.bitwise_or(model_mask,manual_mask))
                    intersection=np.sum(cv2.bitwise_and(model_mask,manual_mask))
                    iou_i=intersection/union*100 if union > 0 else 0.0
                    iou.append(iou_i)
                    print('[INFO] Class %s : IOU = %.2f percent'%(obj_class,float(iou_i)))
                    rowWriter.writerow([file_x,obj_class,iou_i])
        if render:
            cv2.destroyWindow('Validation Window')
        
        if iou:
            iou_np=np.array(iou)
            iou_np=np.nan_to_num(iou_np)
            # Exclude potential outliers (e.g. if only one type of label is present by mistake)
            # Consider a more robust outlier detection or make this configurable if needed.
            if len(iou_np) > 2:
                iou_sort=np.sort(iou_np)
                iou_filt=iou_sort[1:-1] # Basic outlier removal, might need adjustment
            else:
                iou_filt = iou_np

            if len(iou_filt) > 0:
                iou_min=np.min(iou_filt)
                iou_max=np.max(iou_filt)
                iou_mean=np.mean(iou_filt)
                print('[INFO] Mean IOU = %.2f percent' % iou_mean)
                print('[INFO] Max IOU = %.2f percent' % iou_max)
                print('[INFO] Min IOU = %.2f percent' % iou_min)
            else:
                print('[INFO] Not enough IOU values to calculate robust statistics after filtering.')
        else:
            print('[INFO] Target label not found in the input files or no overlapping images/labels.')
    
if __name__== "__main__":
    ap=argparse.ArgumentParser(description="Calculates IoU from ground truth and model prediction CSV files.")
    ap.add_argument('--manual_csv',required=True,help='manual label .csv file')
    ap.add_argument('--model_csv',required=True,help='model output .csv file')
    ap.add_argument('--data_dir',required=True,help='path of data directory')
    ap.add_argument('--labels',default=None,help='comma seperated list of labels.')
    ap.add_argument('--output_dir',required=True,help='output dir')
    ap.add_argument('--render',dest="render", action='store_true', help='Option to display annotated images during processing.')
    ap.set_defaults(render=False)

    args = vars(ap.parse_args())
    render = args['render']

    main(args['model_csv'],args['manual_csv'],args['data_dir'],args['labels'],args['output_dir'],render)
