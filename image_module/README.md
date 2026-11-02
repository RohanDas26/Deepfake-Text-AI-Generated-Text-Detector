# Secondary Module: Image Deepfake Detection

> **Note**: This is an isolated, secondary exploration module.  
> The **primary focus** of this project is the **Multi-Modal Text Deepfake & AI-Generated Text Detection System** (DeBERTa v3, DistilGPT-2, Random Forest, Celery MapReduce).

### Module Architecture:
- `image_engine.py`: EfficientNet-B0 CNN + 2D FFT Frequency Analysis + Grad-CAM XAI.
- `train_image.py`: PyTorch training script with data augmentation, Focal Loss, and MLflow logging.

### Isolation Policy:
- Keep this directory completely standalone.
- Do not import image modules or vision dependencies into core text files (`model.py`, `nlp_engine.py`, `tasks.py`, `train.py`).
