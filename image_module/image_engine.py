"""
=============================================================================
IMAGE DEEPFAKE DETECTION MODULE
=============================================================================

Subject Coverage:
  PDNN  - EfficientNet-B0 CNN, Convolutional layers, Global Average Pooling,
          Transfer Learning, Grad-CAM explainability, FFT frequency analysis
  MLOps - MLflow logging of image predictions
  NLP   - Feature extraction pipeline (face detection → feature extraction)
  CN    - HTTP multipart file upload endpoint

Architecture:
  Input Image
      |
      v
  Face Detection (OpenCV Haar Cascade) ← preprocessing pipeline
      |
      v
  Branch A: EfficientNet-B0 CNN       ← PDNN: spatial feature extraction
  Branch B: FFT Frequency Analysis    ← PDNN: frequency domain artifacts
      |
      v
  Fusion + Classification Head        ← PDNN: multi-modal fusion
      |
      v
  REAL / FAKE + Confidence Score
      |
      v
  Grad-CAM Heatmap                   ← PDNN: XAI / explainability
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import io
import base64
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# PDNN CONCEPT: CNN Architecture using EfficientNet-B0
# Convolutional Neural Networks (CNNs) are the backbone of computer vision.
# Key PDNN concepts demonstrated:
#   - Convolution operation (feature map extraction)
#   - Pooling (spatial downsampling)
#   - Transfer Learning (pretrained ImageNet weights)
#   - Global Average Pooling (instead of flatten to reduce parameters)
# =============================================================================
class ImageDeepfakeDetector(nn.Module):
    """
    Dual-branch deepfake image detector:
      Branch A: EfficientNet-B0 CNN — extracts spatial features (faces, textures)
      Branch B: FFT Frequency head  — detects compression/GAN artifacts in freq domain
      Fusion: Weighted combination for final REAL/FAKE classification

    PDNN Concepts:
      - Convolutional layers (learned spatial filters)
      - Batch Normalization (training stability)
      - Dropout (regularization)
      - GELU activation (modern activation function)
      - Transfer Learning (frozen EfficientNet backbone)
      - Global Average Pooling (dimensionality reduction)
    """

    def __init__(self, num_classes: int = 2):
        super(ImageDeepfakeDetector, self).__init__()

        # ----------------------------------------------------------------
        # Branch A: EfficientNet-B0 CNN (Transfer Learning)
        # PDNN: Pretrained CNN backbone — learns hierarchical visual features
        # Layer 1-3: edges, textures
        # Layer 4-6: shapes, faces
        # Layer 7+:  high-level semantic features
        # ----------------------------------------------------------------
        try:
            from torchvision import models
            try:
                efficientnet = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
                )
            except Exception as e:
                logger.warning(f"Could not download ImageNet weights ({e}). Initializing EfficientNet-B0 with unweighted backbone.")
                efficientnet = models.efficientnet_b0(weights=None)

            # Remove the final classification layer (we replace it)
            self.cnn_backbone = nn.Sequential(*list(efficientnet.children())[:-1])

            # PDNN: Freeze backbone layers (Transfer Learning)
            # Only train the classification head on our deepfake data
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False

            cnn_out_dim = 1280  # EfficientNet-B0 output dimension
            self.use_efficientnet = True
            logger.info("EfficientNet-B0 backbone loaded (Transfer Learning)")

        except Exception as e:
            # Fallback: simple CNN if torchvision fails
            logger.warning(f"EfficientNet setup error: {e}. Using simple CNN fallback.")
            self.use_efficientnet = False
            self.cnn_backbone = nn.Sequential(
                # PDNN: Conv2d — the fundamental CNN operation
                # kernel_size=3: 3x3 filter slides over the image
                # stride=2: downsample by 2x each conv
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # PDNN: Global Average Pooling — reduces each feature map to 1 value
                # More parameter-efficient than Flatten
                nn.AdaptiveAvgPool2d((1, 1))
            )
            cnn_out_dim = 256
            self.use_efficientnet = False

        # ----------------------------------------------------------------
        # Branch B: Frequency Analysis Head
        # PDNN: Signal processing meets deep learning
        # FFT (Fast Fourier Transform) converts spatial domain -> frequency domain
        # GAN-generated images leave characteristic frequency artifacts
        # that are invisible to the human eye but detectable by CNNs
        # ----------------------------------------------------------------
        self.freq_head = nn.Sequential(
            nn.Linear(256, 128),      # 256 = 16x16 FFT magnitude bins
            nn.BatchNorm1d(128),       # PDNN: Batch Normalization
            nn.GELU(),                 # PDNN: GELU activation
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # ----------------------------------------------------------------
        # Fusion + Classification Head
        # PDNN: Multi-modal feature fusion
        # Combines CNN spatial features with frequency domain features
        # ----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, num_classes)    # Output: [REAL, FAKE]
        )

    def extract_fft_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        PDNN Concept: Frequency Domain Analysis via 2D FFT.

        How it works:
          1. Convert image to grayscale
          2. Apply 2D Fast Fourier Transform (FFT)
          3. Take the magnitude spectrum |F(u,v)|
          4. Log-scale and flatten into feature vector

        Why: GAN-generated (deepfake) images have characteristic
        frequency artifacts from their upsampling/deconvolution layers.
        These artifacts are invisible in pixel space but visible in
        frequency space.

        Reference: "Thinking in Frequency: Face Forgery Detection by
        Mining Frequency-aware Clues" (Li et al., 2021)
        """
        # Convert to grayscale (average RGB channels)
        gray = image_tensor.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 2D FFT — decompose into frequency components
        fft_out = torch.fft.fft2(gray)

        # Magnitude spectrum (shift zero-frequency to center)
        fft_shifted = torch.fft.fftshift(fft_out, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)

        # Log scale (prevents large values from dominating)
        log_magnitude = torch.log1p(magnitude)

        # Resize to 16x16 and flatten -> 256-dim feature vector
        log_magnitude_resized = F.adaptive_avg_pool2d(log_magnitude, (16, 16))
        freq_features = log_magnitude_resized.squeeze(1).reshape(image_tensor.size(0), -1)

        return freq_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch A: CNN spatial features
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.flatten(1)  # Flatten [B, C, 1, 1] -> [B, C]

        # Branch B: FFT frequency features
        freq_raw = self.extract_fft_features(x)
        freq_features = self.freq_head(freq_raw)

        # Fusion: concatenate both feature branches
        fused = torch.cat([cnn_features, freq_features], dim=1)

        # Classification
        logits = self.classifier(fused)
        return logits


# =============================================================================
# PDNN CONCEPT: Grad-CAM Explainability (XAI)
# Gradient-weighted Class Activation Mapping
# Shows WHICH parts of the image drove the "FAKE" decision
# =============================================================================
def compute_gradcam(model: ImageDeepfakeDetector,
                    image_tensor: torch.Tensor,
                    target_class: int = 1) -> np.ndarray:
    """
    PDNN Concept: Gradient-weighted Class Activation Mapping (Grad-CAM).

    Algorithm:
      1. Forward pass through CNN up to the last conv layer
      2. Backpropagate the score of the target class (FAKE=1)
      3. Compute gradient of target class w.r.t. last conv feature map
      4. Global average pool the gradients -> channel weights
      5. Weighted sum of feature maps -> heatmap
      6. ReLU the heatmap (keep only positive activations)

    Result: A heatmap showing which facial regions contributed most
    to the FAKE prediction. E.g., it might highlight the eyes or
    mouth — common GAN artifact locations.

    Reference: "Grad-CAM: Visual Explanations from Deep Networks"
    (Selvaraju et al., 2017)
    """
    model.eval()
    gradients = []
    activations = []

    # Register hooks to capture gradients and activations
    def save_gradient(grad):
        gradients.append(grad.detach().cpu())

    def save_activation(module, input, output):
        activations.append(output.detach())
        output.register_hook(save_gradient)

    # Find the last conv layer in the CNN backbone
    target_layer = None
    for module in model.cnn_backbone.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            target_layer = module

    if target_layer is None:
        return np.zeros((224, 224))

    hook = target_layer.register_forward_hook(save_activation)

    # Enable gradients for XAI
    image_tensor.requires_grad_(True)
    model.zero_grad()

    try:
        logits = model(image_tensor)
        score = logits[0, target_class]
        score.backward()

        if not gradients or not activations:
            return np.zeros((224, 224))

        # Compute Grad-CAM
        grads = gradients[0]      # [1, C, H, W]
        acts = activations[0]     # [1, C, H, W]

        # Global average pool gradients -> weights (one per channel)
        weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # Keep only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        return np.zeros((8, 8))
    finally:
        hook.remove()
        model.zero_grad()


# =============================================================================
# PREPROCESSING: Face Detection + Image Preparation
# =============================================================================
def preprocess_image(image_bytes: bytes,
                     target_size: int = 224) -> tuple:
    """
    Image preprocessing pipeline:
      1. Load image from bytes
      2. Detect face using OpenCV Haar Cascade (or full image fallback)
      3. Crop and resize face region to target_size x target_size
      4. Normalize with ImageNet mean/std (required for EfficientNet)

    Returns:
      - tensor: preprocessed image tensor [1, 3, H, W]
      - face_detected: bool
      - face_coords: (x, y, w, h) or None
    """
    from PIL import Image
    import torchvision.transforms as T

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size

    face_detected = False
    face_coords = None

    # Try face detection with OpenCV
    try:
        import cv2
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            # Use the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Add 20% padding around face
            pad = int(0.2 * min(w, h))
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img_np.shape[1] - x, w + 2 * pad)
            h = min(img_np.shape[0] - y, h + 2 * pad)
            img = img.crop((x, y, x + w, y + h))
            face_detected = True
            face_coords = (int(x), int(y), int(w), int(h))

    except Exception as e:
        logger.warning(f"Face detection bypassed or failed ({e}). Using full image for analysis.")

    # Standard ImageNet normalization transform (required for EfficientNet)
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225]      # ImageNet std
        )
    ])

    tensor = transform(img).unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
    return tensor, face_detected, face_coords, original_size


# =============================================================================
# INFERENCE: Full image deepfake detection pipeline
# =============================================================================
_image_model = None

def get_image_model() -> ImageDeepfakeDetector:
    """Lazy load singleton image deepfake model."""
    global _image_model
    if _image_model is None:
        import os
        logger.info("Loading ImageDeepfakeDetector...")
        _image_model = ImageDeepfakeDetector(num_classes=2)

        weight_path = "image_model.pt" if os.path.exists("image_model.pt") else (
            "image_module/image_model.pt" if os.path.exists("image_module/image_model.pt") else None
        )
        if weight_path:
            logger.info(f"Loading pretrained {weight_path} weights...")
            _image_model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        else:
            logger.warning("image_model.pt not found. Using untrained model (for demo).")

        _image_model.eval()
    return _image_model


def run_image_inference(image_bytes: bytes) -> dict:
    """
    Full image deepfake detection pipeline:
      1. Preprocess image (face detection + normalization)
      2. CNN + FFT dual-branch forward pass
      3. Softmax probabilities
      4. Grad-CAM heatmap generation
      5. MLflow logging

    Returns:
      dict with label, confidence, face detection info,
      model scores, and base64-encoded Grad-CAM heatmap
    """
    import time
    start = time.perf_counter()

    model = get_image_model()

    # Step 1: Preprocess
    try:
        image_tensor, face_detected, face_coords, original_size = preprocess_image(image_bytes)
    except Exception as e:
        return {"error": f"Image preprocessing failed: {str(e)}"}

    # Step 2: Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = "REAL" if pred_idx == 0 else "FAKE (DEEPFAKE)"
    confidence = float(np.max(probs))

    # Step 3: Grad-CAM heatmap
    gradcam_b64 = None
    try:
        # Re-enable grad for Grad-CAM
        cam = compute_gradcam(model, image_tensor.clone(), target_class=1)
        # Convert heatmap to colored image (PIL)
        cam_uint8 = (cam * 255).astype(np.uint8)
        if cam_uint8.ndim == 0 or cam_uint8.size == 1:
            cam_uint8 = np.zeros((8, 8), dtype=np.uint8)
        cam_img = Image.fromarray(cam_uint8, mode="L").resize((224, 224))
        buf = io.BytesIO()
        cam_img.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    result = {
        "label": label,
        "confidence": round(confidence, 4),
        "real_probability": round(float(probs[0]), 4),
        "fake_probability": round(float(probs[1]), 4),
        "face_detected": face_detected,
        "face_coords": face_coords,
        "original_image_size": original_size,
        "model": "EfficientNet-B0 CNN + FFT Frequency Analysis",
        "xai": {
            "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
            "gradcam_heatmap_base64": gradcam_b64
        },
        "performance": {
            "latency_ms": round(latency_ms, 2)
        },
        "pdnn_features_used": [
            "EfficientNet-B0 CNN (Transfer Learning)",
            "2D FFT Frequency Domain Analysis",
            "Batch Normalization",
            "GELU Activation",
            "Global Average Pooling",
            "Grad-CAM XAI"
        ]
    }

    # MLflow logging (MLOps)
    try:
        import mlflow
        import os
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("Image_Deepfake_Detection")
        with mlflow.start_run():
            mlflow.log_metric("image_confidence", confidence)
            mlflow.log_metric("image_latency_ms", latency_ms)
            mlflow.log_param("image_prediction", label)
            mlflow.log_param("face_detected", face_detected)
    except Exception:
        pass

    return result
