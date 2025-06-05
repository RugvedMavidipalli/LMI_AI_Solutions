# OCR Models

This package is a placeholder for Optical Character Recognition models.
Currently a `PaddleOCR` submodule is expected via git submodule.

Example usage once the submodule is populated:
```python
from ocr_models.PaddleOCR import PaddleOCR
ocr = PaddleOCR()
result = ocr.ocr('text_image.png')
```
