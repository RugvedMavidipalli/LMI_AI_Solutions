# Anomaly Detectors

Modules related to visual anomaly detection.

## Subpackages
- `anomalib_lmi` – Adapters and utilities built on top of the [Anomalib](https://github.com/lmitechnologies/anomalib) library.
- `ad_core` – Base classes and registry for anomaly detector wrappers.
- `legacy` – Older implementations kept for reference.

## Core API
`ad_core` exposes a registry pattern similar to the object detector package.

```python
from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector

metadata = {
    'framework': 'anomalib',
    'model_name': 'stfpm',
    'task': 'seg',
    'version': 'v1',
    'model_path': 'weights/model.ckpt'
}

model = AnomalyDetector(metadata)
result = model.predict('image.png')
```

Detailed tutorials for training and conversion to TorchScript are located in `anomalib_lmi/README.md`.
