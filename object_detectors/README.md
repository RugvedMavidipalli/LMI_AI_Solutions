# Object Detectors

Wrappers and utilities for running object detection models.

## Structure
- `od_core` – Base classes (`ODBase`) and `ObjectDetectorRegistry` used to construct detectors from metadata.
- `yolov8_lmi`, `yolov5_lmi` – Implementations for Ultralytics YOLO families with helper scripts to run training and export.
- `detectron2_lmi`, `tf_objdet` – Additional framework adapters.
- `gofactory` – Factory scripts such as `write_validation_json.py` for dataset evaluation.

## Creating a Detector
```python
from object_detectors.od_core.object_detector import ObjectDetector

metadata = {
    'framework': 'ultralytics',
    'model_name': 'yolov8',
    'task': 'od',
    'version': 'v0',
    'model_path': 'weights/best.pt'
}

model = ObjectDetector(metadata)
results, times = model.predict(image, conf=0.5)
```

Scripts in each subfolder can be executed directly. For example to run YOLOv8 training inside Docker:
```
python -m object_detectors.yolov8_lmi.run_cmd
```
See the sub‑package READMEs for step‑by‑step instructions.
