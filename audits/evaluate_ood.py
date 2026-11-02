import json, pickle, torch, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from math import sqrt
from collections import defaultdict

def wilson_interval(k, n, confidence=0.95):
    if n == 0: return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

with open('dataset/curated/raid_ood_benchmark.json', 'r', encoding='utf-8') as f:
    ood = json.load(f)
with open('dataset/curated/ood_features.pkl', 'rb') as f:
    ood_feats = pickle.load(f)

texts = [x['text'] for x in ood]
y_true = np.array([x['label'] for x in ood])

from model import PDNCDualBranchNetwork, get_tokenizer
tokenizer = get_tokenizer()
model = PDNCDualBranchNetwork(stat_dim=8, num_classes=3)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

with open('rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

# Mini-batched inference (bsz=16)
bsz = 16
all_probs_dl = []
for i in range(0, len(texts), bsz):
    b_texts = texts[i:i+bsz]
    b_stats = torch.tensor(ood_feats[i:i+bsz], dtype=torch.float32)
    enc = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'], b_stats)
        probs = F.softmax(logits, dim=1).numpy()
        all_probs_dl.append(probs)

probs_dl = np.vstack(all_probs_dl)
probs_rf = rf.predict_proba(ood_feats)

# 1. Pure DeBERTa
pred_deb = np.argmax(probs_dl, axis=1)
acc_deb = accuracy_score(y_true, pred_deb)

# 2. Pure Random Forest
pred_rf = np.argmax(probs_rf, axis=1)
acc_rf = accuracy_score(y_true, pred_rf)

# 3. Raw Ensemble (No PPL Rule)
raw_ens = 0.65 * probs_dl + 0.35 * probs_rf
pred_raw = np.argmax(raw_ens, axis=1)
acc_raw = accuracy_score(y_true, pred_raw)

# 4. Calibrated Ensemble (With PPL Rule)
cal_ens = np.copy(raw_ens)
for i in range(len(cal_ens)):
    ppl = np.exp(ood_feats[i, 0])
    if ppl < 28.0 and cal_ens[i, 0] > 0.35:
        transfer = cal_ens[i, 0] * 0.70
        cal_ens[i, 0] -= transfer
        cal_ens[i, 1] += transfer * 0.60
        cal_ens[i, 2] += transfer * 0.40
        cal_ens[i] = cal_ens[i] / np.sum(cal_ens[i])

pred_cal = np.argmax(cal_ens, axis=1)
acc_cal = accuracy_score(y_true, pred_cal)
low_c, high_c = wilson_interval(np.sum(pred_cal == y_true), len(y_true))

print('\n===========================================================================')
print('   EMPIRICAL OUT-OF-DISTRIBUTION (OOD) EVALUATION (102 UNSEEN SAMPLES)   ')
print('===========================================================================')
print(f'DeBERTa Alone Accuracy:         {acc_deb*100:.1f}% ({np.sum(pred_deb == y_true)}/102)')
print(f'Random Forest Alone Accuracy:   {acc_rf*100:.1f}% ({np.sum(pred_rf == y_true)}/102)')
print(f'Raw Ensemble (No PPL Rule):     {acc_raw*100:.1f}% ({np.sum(pred_raw == y_true)}/102)')
print(f'Final Ensemble (With PPL Rule): {acc_cal*100:.1f}% ({np.sum(pred_cal == y_true)}/102) [95% CI: {low_c*100:.1f}% - {high_c*100:.1f}%]')

print('\n--- Confusion Matrix (Final Ensemble) ---')
cm = confusion_matrix(y_true, pred_cal)
names = ['HUMAN', 'AI', 'PARAPHRASED']
print(f'{"True Class":14s} {"Pred HUMAN":12s} {"Pred AI":10s} {"Pred PARAPHRASED":18s} {"Recall"}')
for i in range(3):
    rec = cm[i, i] / np.sum(cm[i]) * 100
    print(f'{names[i]:14s} {cm[i,0]:10d} {cm[i,1]:10d} {cm[i,2]:18d}   {rec:.1f}%')

print('\n--- Performance Breakdown by Unseen Domain ---')
dom_correct = defaultdict(int)
dom_total = defaultdict(int)
for i, x in enumerate(ood):
    d = x.get('domain', 'unknown')
    dom_total[d] += 1
    if pred_cal[i] == y_true[i]:
        dom_correct[d] += 1

for d in sorted(dom_total.keys()):
    c = dom_correct[d]
    t = dom_total[d]
    pct = c / t * 100
    print(f'  Domain: {d:12s} -> {c}/{t} ({pct:.1f}%)')

print('\n--- Performance Breakdown by Generator ---')
gen_correct = defaultdict(int)
gen_total = defaultdict(int)
for i, x in enumerate(ood):
    g = x.get('model', 'unknown')
    gen_total[g] += 1
    if pred_cal[i] == y_true[i]:
        gen_correct[g] += 1

for g in sorted(gen_total.keys()):
    c = gen_correct[g]
    t = gen_total[g]
    pct = c / t * 100
    print(f'  Generator: {g:14s} -> {c}/{t} ({pct:.1f}%)')
print('===========================================================================\n')
