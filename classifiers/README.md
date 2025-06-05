# Classifiers

Package providing image classification models.

## Available Modules
- `yolov8_cls` – Wrapper around Ultralytics YOLOv8 for image classification.
- `efficientnet` – (empty placeholder for EfficientNet based models).

## YOLOv8 Classification
The `yolov8_cls` module contains:
- `model.py` – `Yolov8_cls` class built on top of the detection model wrapper.
- `run_cmd.py` – Helper script to run training/validation/prediction inside a Docker container using a YAML config.
- `run_model.py` – Small script for running inference from the command line.

### Example
```python
from classifiers.yolov8_cls.model import Yolov8_cls

model = Yolov8_cls('weights/best.pt', device='gpu')
result, _ = model.predict('image.jpg')
print(result)
```

See `yolov8_cls/README.md` for a detailed training walkthrough.
