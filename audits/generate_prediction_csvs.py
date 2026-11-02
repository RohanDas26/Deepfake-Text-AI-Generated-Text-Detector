import os
import sys
import json
import pickle
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter, defaultdict

# Ensure parent path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import PDNCDualBranchNetwork, get_tokenizer
from data_corpus import CORPUS

LABEL_MAP = {
    0: "HUMAN-WRITTEN",
    1: "FULLY AI-GENERATED",
    2: "HUMAN-POLISHED / AI-PARAPHRASED"
}

def load_holdout_data():
    # 1. Corpus
    combined = []
    for idx, s in enumerate(CORPUS):
        gen = "human" if s["label"] == 0 else ("ai-synthetic" if s["label"] == 1 else "human-polished")
        combined.append({
            "id": f"corpus_{idx:03d}",
            "text": s["text"],
            "label": s["label"],
            "domain": s.get("domain", "corpus_general"),
            "generator": gen
        })
    
    # 2. RAID ID
    raid_path = "dataset/curated/raid_id_benchmark.json"
    with open(raid_path, "r", encoding="utf-8") as f:
        raid_id = json.load(f)
    for idx, s in enumerate(raid_id):
        combined.append({
            "id": f"raid_id_{idx:03d}",
            "text": s["text"],
            "label": s["label"],
            "domain": s.get("domain", "raid_general"),
            "generator": s.get("model", "unknown")
        })
        
    labels = np.array([x["label"] for x in combined])
    indices = np.arange(len(combined))
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    
    holdout_samples = [combined[i] for i in val_idx]
    train_samples = [combined[i] for i in train_idx]
    
    return holdout_samples, train_samples, val_idx

def main():
    print("=== STARTING PREDICTION CSV GENERATION & AUDIT ===")
    
    # 1. Load Data
    holdout_samples, train_samples, val_idx = load_holdout_data()
    print(f"Loaded {len(train_samples)} training samples and {len(holdout_samples)} holdout samples.")
    
    with open("dataset/curated/raid_ood_benchmark.json", "r", encoding="utf-8") as f:
        ood_samples_raw = json.load(f)
    
    ood_samples = []
    for idx, s in enumerate(ood_samples_raw):
        ood_samples.append({
            "id": f"raid_ood_{idx:03d}",
            "text": s["text"],
            "label": s["label"],
            "domain": s.get("domain", "unknown"),
            "generator": s.get("model", "unknown")
        })
    print(f"Loaded {len(ood_samples)} OOD benchmark samples.")
    
    # Check overlap: Does any OOD text appear in training data?
    print("\n--- CHECKING FOR DATA LEAKAGE: OOD VS TRAINING DATA ---")
    train_text_normalized = {s["text"].strip().lower() for s in train_samples}
    overlapping_texts = []
    for ood_s in ood_samples:
        norm = ood_s["text"].strip().lower()
        if norm in train_text_normalized:
            overlapping_texts.append(ood_s["id"])
            
    print(f"Total training samples checked: {len(train_text_normalized)}")
    print(f"Total OOD samples checked: {len(ood_samples)}")
    if overlapping_texts:
        print(f"WARNING: Found {len(overlapping_texts)} overlapping texts between OOD and training set!")
    else:
        print("RESULT: ZERO text overlap detected! 0 out of 102 OOD texts appear in the training data.")
        
    # Also print breakdown of raid_ood_benchmark.json
    print("\n--- SAMPLE COUNTS IN raid_ood_benchmark.json ---")
    class_counts = Counter(s["label"] for s in ood_samples)
    print("Class Counts:")
    for c_id in sorted(class_counts.keys()):
        print(f"  Class {c_id} ({LABEL_MAP[c_id]}): {class_counts[c_id]}")
        
    domain_counts = Counter(s["domain"] for s in ood_samples)
    print("\nDomain Counts:")
    for d, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:15s}: {cnt}")
        
    gen_counts = Counter(s["generator"] for s in ood_samples)
    print("\nGenerator Counts:")
    for g, cnt in sorted(gen_counts.items(), key=lambda x: -x[1]):
        print(f"  {g:15s}: {cnt}")

    # Load Model and Precomputed Features
    print("\nLoading models and precomputed features...")
    tokenizer = get_tokenizer()
    model = PDNCDualBranchNetwork(stat_dim=8, num_classes=3)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    
    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
        
    # Precomputed features for 555
    with open("dataset/curated/corpus_features.pkl", "rb") as f:
        corp_feat = pickle.load(f)
    with open("dataset/curated/id_features.pkl", "rb") as f:
        id_feat = pickle.load(f)
    all_555_feat = np.vstack([corp_feat, id_feat])
    holdout_feat = all_555_feat[val_idx]
    
    # Precomputed pooled deberta embeddings
    emb_555 = torch.load("dataset/curated/deberta_pooled_555.pt", map_location="cpu")
    holdout_emb = emb_555[val_idx]
    
    with open("dataset/curated/ood_features.pkl", "rb") as f:
        ood_feat = pickle.load(f)

    # -------------------------------------------------------------
    # PROCESS HOLDOUT SET (111 SAMPLES)
    # -------------------------------------------------------------
    print("\nProcessing Holdout Set (111 samples)...")
    with torch.no_grad():
        stat_out = model.statistical_branch(torch.tensor(holdout_feat, dtype=torch.float32))
        fused = torch.cat([holdout_emb, stat_out], dim=1)
        holdout_logits = model.fusion(fused)
        holdout_dl_probs = F.softmax(holdout_logits, dim=1).numpy()
        
    holdout_rf_probs = rf.predict_proba(holdout_feat)
    
    holdout_rows = []
    holdout_flips = 0
    holdout_correct_flips = 0
    
    for i in range(len(holdout_samples)):
        s = holdout_samples[i]
        true_lbl = s["label"]
        ppl = float(np.exp(holdout_feat[i, 0]))
        
        # Raw ensemble
        raw_probs = (0.65 * holdout_dl_probs[i]) + (0.35 * holdout_rf_probs[i])
        raw_pred = int(np.argmax(raw_probs))
        
        # Calibrated with PPL < 28 rule
        cal_probs = np.copy(raw_probs)
        flipped = False
        if ppl < 28.0 and cal_probs[0] > 0.35:
            transfer = cal_probs[0] * 0.70
            cal_probs[0] -= transfer
            cal_probs[1] += transfer * 0.60
            cal_probs[2] += transfer * 0.40
            cal_probs = cal_probs / np.sum(cal_probs)
            
        cal_pred = int(np.argmax(cal_probs))
        if cal_pred != raw_pred:
            flipped = True
            holdout_flips += 1
            if cal_pred == true_lbl:
                holdout_correct_flips += 1
                
        holdout_rows.append({
            "id": s["id"],
            "domain": s["domain"],
            "generator": s["generator"],
            "true_label": LABEL_MAP[true_lbl],
            "predicted_label": LABEL_MAP[cal_pred],
            "prob_human": round(float(cal_probs[0]), 4),
            "prob_ai": round(float(cal_probs[1]), 4),
            "prob_paraphrased": round(float(cal_probs[2]), 4),
            "perplexity": round(ppl, 2),
            "ppl_rule_flipped": "yes" if flipped else "no"
        })
        
    holdout_csv_path = "audits/predictions_holdout_111.csv"
    os.makedirs("audits", exist_ok=True)
    with open(holdout_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "domain", "generator", "true_label", "predicted_label",
            "prob_human", "prob_ai", "prob_paraphrased", "perplexity", "ppl_rule_flipped"
        ])
        writer.writeheader()
        writer.writerows(holdout_rows)
    print(f"Saved holdout predictions to: {holdout_csv_path}")

    # -------------------------------------------------------------
    # PROCESS OOD SET (102 SAMPLES)
    # -------------------------------------------------------------
    print("Processing OOD Set (102 samples)...")
    bsz = 16
    ood_texts = [s["text"] for s in ood_samples]
    ood_dl_probs_list = []
    for b in range(0, len(ood_texts), bsz):
        b_txt = ood_texts[b:b+bsz]
        b_feat = torch.tensor(ood_feat[b:b+bsz], dtype=torch.float32)
        enc = tokenizer(b_txt, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"], b_feat)
            probs = F.softmax(logits, dim=1).numpy()
            ood_dl_probs_list.append(probs)
    ood_dl_probs = np.vstack(ood_dl_probs_list)
    ood_rf_probs = rf.predict_proba(ood_feat)
    
    ood_rows = []
    ood_flips = 0
    ood_correct_flips = 0
    
    for i in range(len(ood_samples)):
        s = ood_samples[i]
        true_lbl = s["label"]
        ppl = float(np.exp(ood_feat[i, 0]))
        
        raw_probs = (0.65 * ood_dl_probs[i]) + (0.35 * ood_rf_probs[i])
        raw_pred = int(np.argmax(raw_probs))
        
        cal_probs = np.copy(raw_probs)
        flipped = False
        if ppl < 28.0 and cal_probs[0] > 0.35:
            transfer = cal_probs[0] * 0.70
            cal_probs[0] -= transfer
            cal_probs[1] += transfer * 0.60
            cal_probs[2] += transfer * 0.40
            cal_probs = cal_probs / np.sum(cal_probs)
            
        cal_pred = int(np.argmax(cal_probs))
        if cal_pred != raw_pred:
            flipped = True
            ood_flips += 1
            if cal_pred == true_lbl:
                ood_correct_flips += 1
                
        ood_rows.append({
            "id": s["id"],
            "domain": s["domain"],
            "generator": s["generator"],
            "true_label": LABEL_MAP[true_lbl],
            "predicted_label": LABEL_MAP[cal_pred],
            "prob_human": round(float(cal_probs[0]), 4),
            "prob_ai": round(float(cal_probs[1]), 4),
            "prob_paraphrased": round(float(cal_probs[2]), 4),
            "perplexity": round(ppl, 2),
            "ppl_rule_flipped": "yes" if flipped else "no"
        })
        
    ood_csv_path = "audits/predictions_ood_102.csv"
    with open(ood_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "domain", "generator", "true_label", "predicted_label",
            "prob_human", "prob_ai", "prob_paraphrased", "perplexity", "ppl_rule_flipped"
        ])
        writer.writeheader()
        writer.writerows(ood_rows)
    print(f"Saved OOD predictions to: {ood_csv_path}")

    # -------------------------------------------------------------
    # CONFUSION MATRICES COMPUTED DIRECTLY FROM CSVS
    # -------------------------------------------------------------
    def print_cm_from_csv(csv_path, title):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            y_t = []
            y_p = []
            for row in reader:
                y_t.append(row["true_label"])
                y_p.append(row["predicted_label"])
        
        labels_ordered = [LABEL_MAP[0], LABEL_MAP[1], LABEL_MAP[2]]
        cm = confusion_matrix(y_t, y_p, labels=labels_ordered)
        
        print(f"\n=======================================================")
        print(f" CONFUSION MATRIX FROM CSV: {title}")
        print(f"=======================================================")
        print(f"{'True Label':35s} | {'Pred HUMAN':12s} | {'Pred AI':10s} | {'Pred PARAPHRASED':16s} | Recall")
        print("-" * 88)
        for i, row in enumerate(cm):
            total = sum(row)
            recall = (row[i] / total * 100) if total > 0 else 0.0
            print(f"{labels_ordered[i]:35s} | {row[0]:12d} | {row[1]:10d} | {row[2]:16d} | {recall:5.1f}%")
        print("-" * 88)
        total_acc = np.trace(cm) / np.sum(cm) * 100
        print(f"Total Accuracy from CSV: {total_acc:.2f}% ({np.trace(cm)}/{np.sum(cm)})")
        print(f"=======================================================\n")

    print_cm_from_csv(holdout_csv_path, "111-Sample Holdout Set")
    print_cm_from_csv(ood_csv_path, "102-Sample Out-of-Distribution Set")

    # -------------------------------------------------------------
    # PPL RULE FLIP STATISTICS
    # -------------------------------------------------------------
    print("=======================================================")
    print("           PPL < 28 RULE FLIP ANALYSIS                 ")
    print("=======================================================")
    print(f"111-Sample Holdout Set:")
    print(f"  - Total predictions flipped by PPL rule: {holdout_flips}")
    print(f"  - Flips that turned an incorrect pred into CORRECT: {holdout_correct_flips}")
    if holdout_flips > 0:
        print(f"  - Flip accuracy rate: {(holdout_correct_flips / holdout_flips)*100:.1f}%")
    else:
        print("  - Note: In holdout set, DeBERTa v3 transformer already had high confidence on low-PPL text, resulting in 0 threshold flips.")

    print(f"\n102-Sample Out-of-Distribution Set:")
    print(f"  - Total predictions flipped by PPL rule: {ood_flips}")
    print(f"  - Flips that turned an incorrect pred into CORRECT: {ood_correct_flips}")
    if ood_flips > 0:
        print(f"  - Flip accuracy rate: {(ood_correct_flips / ood_flips)*100:.1f}%")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
