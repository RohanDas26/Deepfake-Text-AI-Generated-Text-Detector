"""
=============================================================================
DEEP INTENSE TRAINING PIPELINE — PDNCDualBranchNetwork & Random Forest
=============================================================================
Academic Coverage:
  PDNN  - DeBERTa v3 transformer (frozen backbone transfer learning)
        - Statistical branch with BatchNorm1d & GELU activations
        - Multi-modal contextual + statistical fusion head
        - Focal Loss (gamma=2.0) with class weighting for hard negative mining
        - AdamW optimizer with decoupled L2 weight decay (1e-4)
        - CosineAnnealingLR scheduler with smooth convergence
        - Gradient clipping (max_norm=1.0) preventing gradient explosion
        - Stratified validation holdout set evaluation (Accuracy, F1, Confusion Matrix)
  NLP   - Perplexity (DistilGPT-2), Burstiness (sentence length std/mean), TTR
  MLOps - MLflow experiment tracking & model registry logging
        - Ensemble learning: 60% DeBERTa + 40% Random Forest validation
=============================================================================
"""

import os
import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import mlflow

from model import PDNCDualBranchNetwork, get_tokenizer
from nlp_engine import NLPEngine
from data_corpus import CORPUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# PDNN: Focal Loss
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = focal_loss * at

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


def extract_dataset_features(nlp_engine, tokenizer):
    """
    Loads 555 multi-domain samples (195 CORPUS + 360 RAID) and their precomputed 8-dim NLP features.
    """
    import json
    import pickle

    # 1. Load CORPUS (195 samples: 65 Human, 65 AI, 65 Paraphrased)
    combined_samples = [{"text": s["text"], "label": s["label"]} for s in CORPUS]

    # 2. Load RAID ID Benchmark (360 samples: 120 Human, 120 AI, 120 Paraphrased)
    raid_path = "dataset/curated/raid_id_benchmark.json"
    if os.path.exists(raid_path):
        with open(raid_path, "r", encoding="utf-8") as f:
            raid_data = json.load(f)
            combined_samples.extend([{"text": s["text"], "label": s["label"]} for s in raid_data])

    logger.info(f"Loaded {len(combined_samples)} balanced multi-domain samples across all 3 classes.")

    texts = [item["text"] for item in combined_samples]
    labels = [item["label"] for item in combined_samples]

    # 3. Load or compute 8-dim NLP features
    corpus_feat_path = "dataset/curated/corpus_features.pkl"
    raid_feat_path = "dataset/curated/id_features.pkl"

    if os.path.exists(corpus_feat_path) and os.path.exists(raid_feat_path):
        logger.info("Loading precomputed 8-dim NLP features for all 555 samples...")
        with open(corpus_feat_path, "rb") as f:
            c_feats = pickle.load(f)
        with open(raid_feat_path, "rb") as f:
            r_feats = pickle.load(f)
        stats = np.vstack([c_feats, r_feats])
    else:
        logger.info(f"Extracting features for {len(combined_samples)} samples using NLPEngine...")
        stats = []
        for idx, text in enumerate(texts):
            stats.append(nlp_engine.extract_features(text))
        stats = np.array(stats, dtype=np.float32)

    logger.info("Tokenizing texts with DeBERTa v3 tokenizer (max_length=128 for efficient CPU sequence representations)...")
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    stats_tensor = torch.tensor(stats, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return encodings, stats_tensor, labels_tensor, texts


def evaluate_predictions(y_true, y_pred, title="MODEL EVALUATION"):
    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    label_names = ["HUMAN", "AI", "PARAPHRASED"]
    logger.info(f"\n===== {title} =====")
    logger.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    logger.info(f"  Precision : {prec:.4f} (macro)")
    logger.info(f"  Recall    : {rec:.4f} (macro)")
    logger.info(f"  F1-Score  : {f1:.4f} (macro)")
    logger.info("  Confusion Matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        name = label_names[i] if i < len(label_names) else f"Class {i}"
        logger.info(f"    {name:12s}: {row.tolist()}")
    logger.info("==========================================\n")

    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm.tolist()}


def deep_train():
    logger.info("=== INITIALIZING DEEP INTENSE MODEL TRAINING PIPELINE ===")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Deep_Text_Deepfake_Training")

    # 1. Initialize Engines
    nlp_engine = NLPEngine()
    tokenizer = get_tokenizer()
    model = PDNCDualBranchNetwork(stat_dim=8)

    # Freeze DeBERTa backbone
    for param in model.deberta.parameters():
        param.requires_grad = False

    # 2. Extract and Prepare Multi-Domain Dataset (555 samples from CORPUS + RAID)
    encodings, stats_tensor, labels_tensor, texts = extract_dataset_features(nlp_engine, tokenizer)

    # 3. Stratified Train / Validation Split (80% train, 20% holdout validation)
    indices = np.arange(len(labels_tensor))
    y_all = labels_tensor.numpy()
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_all)

    train_stats = stats_tensor[train_idx]
    train_labels = labels_tensor[train_idx]
    train_input_ids = encodings["input_ids"][train_idx]
    train_attention_mask = encodings["attention_mask"][train_idx]

    val_stats = stats_tensor[val_idx]
    val_labels = labels_tensor[val_idx]
    val_input_ids = encodings["input_ids"][val_idx]
    val_attention_mask = encodings["attention_mask"][val_idx]

    logger.info(f"Dataset split: {len(train_idx)} training samples, {len(val_idx)} validation holdout samples.")

    # 4. Precompute DeBERTa mean pooled embeddings with torch.no_grad() in mini-batches
    # Batch size 16 avoids CPU RAM thrashing with DeBERTa v3's 512-token relative position attention
    logger.info("Pre-computing DeBERTa contextual sequence representations in mini-batches (batch_size=16)...")
    def _compute_pooled_batched(input_ids, attention_mask, bsz=16, split_name="Train"):
        chunks = []
        total_batches = (len(input_ids) + bsz - 1) // bsz
        for i in range(0, len(input_ids), bsz):
            batch_num = i // bsz + 1
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"  [{split_name}] Batch {batch_num}/{total_batches} pooled...")
            b_ids = input_ids[i:i+bsz]
            b_mask = attention_mask[i:i+bsz]
            out = model.deberta(input_ids=b_ids, attention_mask=b_mask)
            mask_exp = b_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            p = torch.sum(out.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
            chunks.append(p)
        return torch.cat(chunks, dim=0)

    pooled_cache_file = "dataset/curated/deberta_pooled_555.pt"
    if os.path.exists(pooled_cache_file):
        logger.info(f"Loading cached DeBERTa pooled representations from {pooled_cache_file}...")
        all_pooled = torch.load(pooled_cache_file, map_location="cpu")
    else:
        logger.info("Pre-computing DeBERTa contextual sequence representations in mini-batches (batch_size=16)...")
        model.eval()
        with torch.no_grad():
            all_pooled = _compute_pooled_batched(encodings["input_ids"], encodings["attention_mask"], bsz=16, split_name="Dataset")
        torch.save(all_pooled, pooled_cache_file)
        logger.info(f"Cached DeBERTa embeddings to {pooled_cache_file}")

    train_pooled = all_pooled[train_idx]
    val_pooled = all_pooled[val_idx]

    # 5. Training Setup
    epochs = 70
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=8e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_model_weights = None

    logger.info(f"Starting deep calibration over {epochs} epochs with Focal Loss & CosineAnnealingLR...")

    with mlflow.start_run(run_name="deep_deberta_dual_branch_training"):
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("loss_function", "FocalLoss_gamma2.0")
        mlflow.log_param("num_samples", len(labels_tensor))
        mlflow.log_param("train_samples", len(train_idx))
        mlflow.log_param("val_samples", len(val_idx))

        for epoch in range(epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()

            stat_out = model.statistical_branch(train_stats)
            fused = torch.cat((train_pooled, stat_out), dim=1)
            logits = model.fusion(fused)
            loss = criterion(logits, train_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_stat_out = model.statistical_branch(val_stats)
                val_fused = torch.cat((val_pooled, val_stat_out), dim=1)
                val_logits = model.fusion(val_fused)
                val_loss = criterion(val_logits, val_labels)
                val_preds = torch.argmax(val_logits, dim=1).numpy()
                val_f1 = f1_score(val_labels.numpy(), val_preds, average="macro", zero_division=0)

            mlflow.log_metric("train_loss", float(loss.item()), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss.item()), step=epoch)
            mlflow.log_metric("val_f1", float(val_f1), step=epoch)
            mlflow.log_metric("learning_rate", float(current_lr), step=epoch)

            # Checkpoint best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_weights = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                logger.info(
                    f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss.item():.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}"
                )

        # Load best model weights
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
            logger.info(f"Loaded best checkpoint (Validation F1: {best_val_f1:.4f})")

        # 6. Evaluate Best DeBERTa Model on Holdout Set
        model.eval()
        with torch.no_grad():
            final_stat_out = model.statistical_branch(val_stats)
            final_fused = torch.cat((val_pooled, final_stat_out), dim=1)
            final_logits = model.fusion(final_fused)
            deberta_val_probs = F.softmax(final_logits, dim=1).numpy()
            deberta_val_preds = np.argmax(deberta_val_probs, axis=1)

        deberta_metrics = evaluate_predictions(val_labels.numpy(), deberta_val_preds, title="DEBERTA DUAL-BRANCH VALIDATION RESULTS")
        mlflow.log_metric("deberta_val_acc", deberta_metrics["accuracy"])
        mlflow.log_metric("deberta_val_f1", deberta_metrics["f1"])

        # 7. Train & Calibrate Random Forest on Real Features
        logger.info("Training Random Forest Classifier on real statistical feature vectors...")
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=3, random_state=42)
        rf_clf.fit(train_stats.numpy(), train_labels.numpy())

        rf_val_probs = rf_clf.predict_proba(val_stats.numpy())
        rf_val_preds = np.argmax(rf_val_probs, axis=1)

        rf_metrics = evaluate_predictions(val_labels.numpy(), rf_val_preds, title="RANDOM FOREST VALIDATION RESULTS")
        mlflow.log_metric("rf_val_acc", rf_metrics["accuracy"])
        mlflow.log_metric("rf_val_f1", rf_metrics["f1"])

        # 8. Evaluate Ensemble (65% DeBERTa + 35% RF)
        logger.info("Evaluating Dual-Model Ensemble (65% DeBERTa + 35% Random Forest)...")
        ensemble_val_probs = (0.65 * deberta_val_probs) + (0.35 * rf_val_probs)
        ensemble_val_preds = np.argmax(ensemble_val_probs, axis=1)

        ensemble_metrics = evaluate_predictions(val_labels.numpy(), ensemble_val_preds, title="FINAL ENSEMBLE VALIDATION RESULTS")
        mlflow.log_metric("ensemble_val_acc", ensemble_metrics["accuracy"])
        mlflow.log_metric("ensemble_val_f1", ensemble_metrics["f1"])

        # 9. Save Calibrated Artifacts
        logger.info("Saving trained model weights to model.pt...")
        torch.save(model.state_dict(), "model.pt")

        logger.info("Saving trained Random Forest classifier to rf_model.pkl...")
        with open("rf_model.pkl", "wb") as f:
            pickle.dump(rf_clf, f)

        # 10. Register Model in MLflow Model Registry
        try:
            model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "AITextDetector")
            logger.info("Model registered in MLflow Model Registry as 'AITextDetector'")
        except Exception as e:
            logger.warning(f"Could not register model in MLflow: {e}")

    logger.info("=== DEEP TRAINING & ENSEMBLE CALIBRATION COMPLETED SUCCESSFULLY ===")
    return {
        "deberta_metrics": deberta_metrics,
        "rf_metrics": rf_metrics,
        "ensemble_metrics": ensemble_metrics
    }


if __name__ == "__main__":
    deep_train()
