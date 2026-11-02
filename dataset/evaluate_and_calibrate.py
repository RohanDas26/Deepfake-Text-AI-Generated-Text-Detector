"""
=============================================================================
EVALUATION & CONFIDENCE CALIBRATION PIPELINE
=============================================================================
Academic Focus:
- NLP:   Multi-dimensional feature extraction (Perplexity, Burstiness, TTR,
         Entropy, Sentiment, POS ratios, Named Entities)
- PDNN:  DeBERTa v3 Dual-Branch Architecture, Cross-Entropy / Focal Loss,
         Confidence Calibration (Temperature Scaling & Isotonic Regression)
- MLOps: Group-aware train/val/test splits (source_id isolation),
         Holdout OOD evaluation, ECE (Expected Calibration Error) tracking
=============================================================================
"""

import os
import sys
# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, f1_score, accuracy_score
from scipy.optimize import minimize

from nlp_engine import NLPEngine
from model import PDNCDualBranchNetwork

# Set deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

CLASSES = ["HUMAN-WRITTEN", "FULLY AI-GENERATED", "HUMAN-POLISHED / AI-PARAPHRASED"]

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and returns reliability bin data.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_details = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += bin_error * prop_in_bin
            bin_details.append({
                "range": f"({bin_lower:.1f}, {bin_upper:.1f}]",
                "count": int(np.sum(in_bin)),
                "avg_conf": float(avg_confidence_in_bin),
                "accuracy": float(accuracy_in_bin),
                "error": float(bin_error)
            })
        else:
            bin_details.append({
                "range": f"({bin_lower:.1f}, {bin_upper:.1f}]",
                "count": 0,
                "avg_conf": 0.0,
                "accuracy": 0.0,
                "error": 0.0
            })

    return float(ece), bin_details

def compute_multiclass_brier(probs, labels):
    """Computes mean multiclass Brier score."""
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    for i, l in enumerate(labels):
        one_hot[i, l] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

class TemperatureScaler:
    """Platt scaling extension for multiclass softmax logits."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits, val_labels):
        """Find optimal temperature T > 0 minimizing NLL on validation set."""
        def nll_loss(t):
            temp = t[0]
            if temp <= 0:
                return 1e9
            scaled_logits = val_logits / temp
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            log_likelihood = np.log(np.clip(probs[np.arange(len(val_labels)), val_labels], 1e-12, 1.0))
            return -np.mean(log_likelihood)

        res = minimize(nll_loss, [1.0], method='Nelder-Mead', bounds=[(0.01, 10.0)])
        self.temperature = float(res.x[0])
        return self.temperature

    def transform(self, logits):
        scaled = logits / self.temperature
        exp_logits = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

class IsotonicCalibrator:
    """Non-parametric piecewise constant isotonic calibration per class."""
    def __init__(self):
        self.calibrators = [IsotonicRegression(out_of_bounds='clip') for _ in range(3)]

    def fit(self, val_probs, val_labels):
        for c in range(3):
            binary_target = (val_labels == c).astype(float)
            self.calibrators[c].fit(val_probs[:, c], binary_target)

    def transform(self, probs):
        cal_probs = np.zeros_like(probs)
        for c in range(3):
            cal_probs[:, c] = self.calibrators[c].predict(probs[:, c])
        # Re-normalize across classes
        sums = np.sum(cal_probs, axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return cal_probs / sums

def extract_dataset_features(samples, cache_path="dataset/curated/features_cache.pkl"):
    """Extracts or loads cached NLP linguistic & statistical features."""
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Extracting NLP features for {len(samples)} samples via NLPEngine...", flush=True)
    nlp = NLPEngine()
    features = []
    for idx, s in enumerate(samples):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(samples)} samples...", flush=True)
        feat_vec = nlp.extract_features(s["text"])
        features.append(feat_vec)

    features = np.array(features, dtype=np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Features cached to {cache_path}.", flush=True)
    return features

def run_evaluation_and_calibration():
    # 1. Load Data
    with open("dataset/curated/raid_id_benchmark.json", "r", encoding="utf-8") as f:
        id_data = json.load(f)
    with open("dataset/curated/raid_ood_benchmark.json", "r", encoding="utf-8") as f:
        ood_data = json.load(f)

    print(f"Loaded {len(id_data)} In-Distribution and {len(ood_data)} Out-of-Distribution samples.")

    # 2. Extract Features
    id_features = extract_dataset_features(id_data, "dataset/curated/id_features.pkl")
    ood_features = extract_dataset_features(ood_data, "dataset/curated/ood_features.pkl")

    id_labels = np.array([s["label"] for s in id_data], dtype=np.int64)
    id_sources = np.array([s["source_id"] for s in id_data])

    ood_labels = np.array([s["label"] for s in ood_data], dtype=np.int64)

    # 3. Group-Aware Split on source_id (prevents prompt/topic leakage)
    # 70% Train, 15% Val, 15% Test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(id_data, id_labels, groups=id_sources))

    temp_data = [id_data[i] for i in temp_idx]
    temp_labels = id_labels[temp_idx]
    temp_sources = id_sources[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_sub_idx, test_sub_idx = next(gss_val.split(temp_data, temp_labels, groups=temp_sources))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    # Verify zero leakage across splits
    train_src = set(id_sources[train_idx])
    val_src = set(id_sources[val_idx])
    test_src = set(id_sources[test_idx])
    print(f"\nGroup-Aware Split Verification:")
    print(f"  Train: {len(train_idx)} samples | {len(train_src)} unique sources")
    print(f"  Val:   {len(val_idx)} samples | {len(val_src)} unique sources")
    print(f"  Test:  {len(test_idx)} samples | {len(test_src)} unique sources")
    print(f"  Train-Test source overlap: {len(train_src.intersection(test_src))} (must be 0)")
    print(f"  Train-Val source overlap:  {len(train_src.intersection(val_src))} (must be 0)")

    # 4. Train Classifiers
    # Classifier A: Random Forest on 8 NLP features
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(id_features[train_idx], id_labels[train_idx])

    val_rf_probs = rf.predict_proba(id_features[val_idx])
    test_rf_probs = rf.predict_proba(id_features[test_idx])
    ood_rf_probs = rf.predict_proba(ood_features)

    # Classifier B: Dual-Branch DeBERTa Network Evaluation
    # Load production model weights
    device = torch.device("cpu")
    model = PDNCDualBranchNetwork()
    model.eval()
    if os.path.exists("model.pt"):
        state_dict = torch.load("model.pt", map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded production PDNCDualBranchNetwork weights from model.pt.")

    nlp_engine = NLPEngine()
    
    def get_nn_predictions(samples, split_features, split_name=""):
        all_logits = []
        print(f"Computing DeBERTa logits for {split_name} ({len(samples)} samples)...", flush=True)
        with torch.no_grad():
            for idx, s in enumerate(samples):
                if idx > 0 and idx % 25 == 0:
                    print(f"  {split_name}: {idx}/{len(samples)}...", flush=True)
                tokens = model.tokenizer(
                    s["text"],
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                stat_tensor = torch.tensor([split_features[idx]], dtype=torch.float32)
                logits = model(tokens["input_ids"], tokens["attention_mask"], stat_tensor)
                all_logits.append(logits.squeeze(0).numpy())
        return np.array(all_logits)

    val_nn_logits = get_nn_predictions([id_data[i] for i in val_idx], id_features[val_idx], "Validation")
    test_nn_logits = get_nn_predictions([id_data[i] for i in test_idx], id_features[test_idx], "Test")
    ood_nn_logits = get_nn_predictions(ood_data, ood_features, "OOD")

    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    val_nn_probs = softmax(val_nn_logits)
    test_nn_probs = softmax(test_nn_logits)
    ood_nn_probs = softmax(ood_nn_logits)

    # Classifier C: Ensemble (60% DeBERTa Dual-Branch + 40% Random Forest)
    val_ens_probs = 0.60 * val_nn_probs + 0.40 * val_rf_probs
    test_ens_probs = 0.60 * test_nn_probs + 0.40 * test_rf_probs
    ood_ens_probs = 0.60 * ood_nn_probs + 0.40 * ood_rf_probs

    # 5. Fit Calibrators on Validation Set
    print("\nFitting Confidence Calibrators on Validation Split...")
    # A. Temperature Scaler on NN logits
    temp_scaler = TemperatureScaler()
    opt_temp = temp_scaler.fit(val_nn_logits, id_labels[val_idx])
    print(f"  Optimized Temperature (T): {opt_temp:.4f}")

    # B. Isotonic Calibrator on Ensemble Probabilities
    iso_cal = IsotonicCalibrator()
    iso_cal.fit(val_ens_probs, id_labels[val_idx])
    print("  Fitted 3-class Isotonic Regressors.")

    # 6. Apply Calibration
    # Calibrated probabilities for Test set
    test_nn_temp_probs = temp_scaler.transform(test_nn_logits)
    test_ens_temp_probs = 0.60 * test_nn_temp_probs + 0.40 * test_rf_probs
    test_ens_iso_probs = iso_cal.transform(test_ens_probs)

    # Calibrated probabilities for OOD set
    ood_nn_temp_probs = temp_scaler.transform(ood_nn_logits)
    ood_ens_temp_probs = 0.60 * ood_nn_temp_probs + 0.40 * ood_rf_probs
    ood_ens_iso_probs = iso_cal.transform(ood_ens_probs)

    # 7. Comprehensive Diagnostic Metrics
    def evaluate_model(name, probs, true_labels):
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        ece, bins = compute_ece(probs, true_labels)
        brier = compute_multiclass_brier(probs, true_labels)
        cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
        return {
            "name": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "ece": ece,
            "brier": brier,
            "confusion_matrix": cm.tolist(),
            "bins": bins,
            "preds": preds.tolist()
        }

    results = {}
    print("\n" + "="*70)
    print("IN-DISTRIBUTION TEST EVALUATION (Unseen Prompts & Documents)")
    print("="*70)

    models_to_eval = [
        ("RF Only (NLP Statistical)", test_rf_probs),
        ("DeBERTa Dual-Branch (Uncalibrated)", test_nn_probs),
        ("Ensemble (Uncalibrated)", test_ens_probs),
        ("Ensemble + Temperature Scaling", test_ens_temp_probs),
        ("Ensemble + Isotonic Regression", test_ens_iso_probs)
    ]

    for name, p in models_to_eval:
        res = evaluate_model(name, p, id_labels[test_idx])
        results[name] = res
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {res['accuracy']*100:.2f}%")
        print(f"Macro F1:  {res['macro_f1']*100:.2f}%")
        print(f"ECE:       {res['ece']:.4f} (Expected Calibration Error - lower is better)")
        print(f"Brier:     {res['brier']:.4f} (Mean squared probability error - lower is better)")
        print("Confusion Matrix:")
        print(np.array(res['confusion_matrix']))

    print("\n" + "="*70)
    print("OUT-OF-DISTRIBUTION (OOD) GENERALIZATION TEST (Unseen Models & Domains)")
    print("="*70)

    ood_models_to_eval = [
        ("OOD Ensemble (Uncalibrated)", ood_ens_probs),
        ("OOD Ensemble + Temperature Scaling", ood_ens_temp_probs),
        ("OOD Ensemble + Isotonic Regression", ood_ens_iso_probs)
    ]

    for name, p in ood_models_to_eval:
        res = evaluate_model(name, p, ood_labels)
        results[name] = res
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {res['accuracy']*100:.2f}%")
        print(f"Macro F1:  {res['macro_f1']*100:.2f}%")
        print(f"ECE:       {res['ece']:.4f}")
        print(f"Brier:     {res['brier']:.4f}")
        print("Confusion Matrix:")
        print(np.array(res['confusion_matrix']))

    # Save detailed evaluation audit report
    audit_output = {
        "metadata": {
            "id_total": len(id_data),
            "ood_total": len(ood_data),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "temperature": opt_temp
        },
        "results": results
    }

    os.makedirs("audits", exist_ok=True)
    with open("audits/calibration_and_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_output, f, indent=2)

    print("\n" + "="*70)
    print("Audit report saved to audits/calibration_and_evaluation_report.json")
    print("="*70)
    return audit_output

if __name__ == "__main__":
    run_evaluation_and_calibration()
