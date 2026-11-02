"""
=============================================================================
IMAGE DEEPFAKE DETECTOR — Training Script
=============================================================================

PDNN Concepts demonstrated:
  - CNN Architecture (EfficientNet-B0 Transfer Learning)
  - Data Augmentation (rotation, flip, color jitter)
  - Focal Loss (handles class imbalance REAL vs FAKE)
  - CosineAnnealingLR (learning rate scheduling)
  - Gradient Clipping (prevents exploding gradients)
  - Evaluation: Accuracy, F1, Precision, Recall, ROC-AUC
  - MLflow Model Registry (MLOps)

To train with real data:
  - Download FaceForensics++ dataset (https://github.com/ondyari/FaceForensics)
  - Or use Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
  - Place real images in: data/real/
  - Place fake images in: data/fake/
  - Then run: python train_image.py
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PDNN: Custom Dataset with Data Augmentation
# Data augmentation is a PDNN regularization technique that artificially
# increases training set size by applying random transformations.
# This reduces overfitting.
# =============================================================================
class DeepfakeImageDataset(Dataset):
    """
    Custom PyTorch Dataset for deepfake image classification.

    PDNN Concepts:
      - Dataset abstraction (mini-batching, shuffling)
      - Data Augmentation: random crops, flips, color jitter
        (prevents overfitting by showing model varied versions of each image)
      - ImageNet normalization (required for pretrained CNN backbones)
    """

    def __init__(self, real_dir: str, fake_dir: str, augment: bool = True):
        self.samples = []  # list of (filepath, label)
        self.augment = augment

        # Label 0 = REAL, Label 1 = FAKE
        for img_file in os.listdir(real_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(real_dir, img_file), 0))

        for img_file in os.listdir(fake_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(fake_dir, img_file), 1))

        logger.info(f"Dataset: {sum(1 for _, l in self.samples if l == 0)} real, "
                    f"{sum(1 for _, l in self.samples if l == 1)} fake images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torchvision.transforms as T
        from PIL import Image

        filepath, label = self.samples[idx]
        img = Image.open(filepath).convert("RGB")

        if self.augment:
            # PDNN: Data Augmentation transforms
            transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),                          # Random spatial crop
                T.RandomHorizontalFlip(p=0.5),              # Mirror augmentation
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Color augmentation
                T.RandomRotation(degrees=10),               # Slight rotation
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        return transform(img), label


# =============================================================================
# PDNN: Focal Loss (same as text model — consistent design)
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses training on hard examples (misclassified samples).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal_loss = focal_loss * at

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


def train_image_model(
    real_dir: str = "data/real",
    fake_dir: str = "data/fake",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3
):
    """
    Full training pipeline for the image deepfake detector.

    PDNN Best Practices Applied:
      1. Transfer Learning — EfficientNet-B0 pretrained on ImageNet
      2. Layer Freezing — train only classification head initially
      3. Focal Loss — handles REAL/FAKE class imbalance
      4. CosineAnnealingLR — smooth learning rate decay
      5. Gradient Clipping — prevents exploding gradients
      6. Data Augmentation — reduces overfitting
      7. Evaluation: F1, Precision, Recall, ROC-AUC

    MLOps:
      - All metrics logged to MLflow
      - Best model saved by validation F1 score
      - Model registered in MLflow Registry
    """
    try:
        from image_engine import ImageDeepfakeDetector
    except ImportError:
        from image_module.image_engine import ImageDeepfakeDetector
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    # Check data directories exist
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        logger.error(f"Data directories not found: {real_dir}, {fake_dir}")
        logger.info("To train with real data, place images in data/real/ and data/fake/")
        logger.info("Using synthetic dummy forward pass to validate architecture...")

        # Validate architecture works
        model = ImageDeepfakeDetector(num_classes=2)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        logger.info(f"Architecture validation PASSED. Output shape: {out.shape}")
        return

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Image_Deepfake_Detection_Training")

    # Load full dataset and split train/val
    full_dataset = DeepfakeImageDataset(real_dir, fake_dir, augment=True)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42,
                                          stratify=[full_dataset.samples[i][1] for i in indices])

    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_dataset = DeepfakeImageDataset(real_dir, fake_dir, augment=False)
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_idx),
        batch_size=batch_size, shuffle=False
    )

    # Initialize model
    model = ImageDeepfakeDetector(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Training on: {device}")

    # PDNN: Focal Loss
    criterion = FocalLoss(gamma=2.0)

    # PDNN: AdamW with weight decay
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # PDNN: CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_f1 = 0.0

    with mlflow.start_run(run_name="image_deepfake_efficientnet"):
        mlflow.log_params({
            "model": "EfficientNet-B0 + FFT",
            "optimizer": "AdamW",
            "lr_scheduler": "CosineAnnealingLR",
            "loss": "FocalLoss_gamma2",
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_clip": 1.0,
            "device": str(device)
        })

        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            train_losses = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()

                # PDNN: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # --- Validation Phase ---
            model.eval()
            all_preds, all_labels, all_probs = [], [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
                    all_probs.extend(probs[:, 1])

            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            val_precision = precision_score(all_labels, all_preds, zero_division=0)
            val_recall = recall_score(all_labels, all_preds, zero_division=0)
            val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
            avg_loss = np.mean(train_losses)

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_auc": val_auc,
                "learning_rate": current_lr
            }, step=epoch)

            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                        f"F1: {val_f1:.4f} | AUC: {val_auc:.4f} | LR: {current_lr:.2e}")

            # Save best model by F1 score (MLOps best practice)
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), "image_model.pt")
                logger.info(f"  -> New best model saved (F1: {best_f1:.4f})")

        logger.info(f"Training complete. Best Validation F1: {best_f1:.4f}")

        # MLflow Model Registry
        try:
            run_id = mlflow.last_active_run().info.run_id
            mlflow.register_model(f"runs:/{run_id}/model", "ImageDeepfakeDetector")
            logger.info("Model registered as 'ImageDeepfakeDetector' in MLflow Registry")
        except Exception as e:
            logger.warning(f"Model registry: {e}")


if __name__ == "__main__":
    train_image_model()
