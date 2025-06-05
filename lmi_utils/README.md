# LMI Utils

Utility modules used throughout the AI Solutions repository.

## Subpackages
- `dataset_utils` - Dataset representations and helpers for resizing, padding and manipulating annotation files.
- `image_utils` - Image preprocessing scripts such as resize, pad, tiling and color utilities.
- `label_utils` - Conversion utilities for different label formats (CSV, COCO, YOLO, etc.) and drawing helpers.
- `gadget_utils` - Tools for depth/profile processing and drawing bounding boxes with `pipeline_utils`.
- `eval_utils` - Helper scripts to compute IoU metrics and precision/recall curves.
- `data_utils` - Functions for reorganising datasets, train/val/test splits and parsing factory data layouts.
- `pcl_utils` - Helpers for working with point clouds (`.pcd`/`.npy`) and generating height maps.
- `postprocess_utils` - Post‑processing helpers for object detection outputs.
- `system_utils` - Basic system utilities.

## Example Usage
Create a bounding box and resize it using `dataset_utils`:

```python
from lmi_utils.dataset_utils.representations import Box

# coordinates in pixels
box = Box(10, 20, 50, 60)
box.resize(orig_h=100, orig_w=100, new_h=200, new_w=200)
print(box.to_numpy())
```

Draw a box on an image with `gadget_utils.pipeline_utils`:

```python
import cv2
from lmi_utils.gadget_utils import pipeline_utils

img = cv2.imread('image.png')
box = [10, 20, 50, 60]
annotated = pipeline_utils.plot_one_box(box, img, label='object')
cv2.imwrite('out.png', annotated)
```

Most modules can also be executed as scripts using `python -m <module>`.
