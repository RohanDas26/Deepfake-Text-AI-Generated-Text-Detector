"""
=============================================================================
PIPELINE AUDIT SCRIPT — Deep Classification Reliability Investigation
=============================================================================
Investigates:
1. Feature extraction quality across all 3 classes
2. Model A (DeBERTa) vs Model B (RF) independent evaluation
3. Ensemble logic verification
4. Dataset quality checks (duplicates, domain bias)
5. Confusion matrix + per-class metrics on training corpus
6. Generalization on UNSEEN samples not in corpus
=============================================================================
"""
import os, sys, pickle, logging, re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARNING)  # suppress verbose model loading

sys.path.insert(0, os.path.dirname(__file__))
from data_corpus import CORPUS
from model import get_model, get_tokenizer
from nlp_engine import NLPEngine

# ============================================================
# SECTION 0: Dataset Quality Check
# ============================================================
print("\n" + "="*70)
print("SECTION 0: DATASET QUALITY AUDIT")
print("="*70)

texts = [c["text"] for c in CORPUS]
labels = [c["label"] for c in CORPUS]
domains = [c["domain"] for c in CORPUS]

label_names = {0: "HUMAN", 1: "AI", 2: "PARAPHRASED"}
print(f"Total samples: {len(CORPUS)}")
for lbl, name in label_names.items():
    count = labels.count(lbl)
    print(f"  Class {lbl} ({name}): {count} samples ({count/len(labels)*100:.1f}%)")

# Check for exact duplicates
unique_texts = set(texts)
n_dups = len(texts) - len(unique_texts)
print(f"\nExact text duplicates: {n_dups}")

# Check near-duplicates (same first 60 chars)
prefixes = [t[:60].strip().lower() for t in texts]
dup_prefixes = [p for p in prefixes if prefixes.count(p) > 1]
if dup_prefixes:
    print(f"NEAR-DUPLICATES detected ({len(dup_prefixes)} texts share prefix):")
    seen = set()
    for i, p in enumerate(prefixes):
        if p in seen: continue
        matches = [j for j,q in enumerate(prefixes) if q == p]
        if len(matches) > 1:
            seen.add(p)
            match_labels = [labels[j] for j in matches]
            print(f"  Labels {match_labels}: '{texts[matches[0]][:80]}'")
else:
    print("No near-duplicates detected.")

# Average text length per class
for lbl, name in label_names.items():
    class_texts = [t for t, l in zip(texts, labels) if l == lbl]
    avg_words = np.mean([len(t.split()) for t in class_texts])
    avg_chars = np.mean([len(t) for t in class_texts])
    print(f"\n  {name} avg words: {avg_words:.1f}, avg chars: {avg_chars:.1f}")

# ============================================================
# SECTION 1: Feature Extraction Analysis
# ============================================================
print("\n" + "="*70)
print("SECTION 1: FEATURE EXTRACTION ANALYSIS (NLPEngine)")
print("="*70)

nlp = NLPEngine()
print("NLPEngine loaded.")

all_features = []
for text in texts:
    feats = nlp.extract_features(text)
    all_features.append(feats)
all_features = np.array(all_features)  # shape: (N, 3)

feat_names = ["Perplexity", "Burstiness", "TTR"]
for lbl, name in label_names.items():
    idxs = [i for i, l in enumerate(labels) if l == lbl]
    class_feats = all_features[idxs]
    print(f"\n  {name} (n={len(idxs)}):")
    for j, feat in enumerate(feat_names):
        vals = class_feats[:, j]
        print(f"    {feat:12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")

# Feature separability: check if features actually differ between classes
print("\n  SEPARABILITY CHECK — Are features discriminative?")
from scipy import stats as scipy_stats
for j, feat in enumerate(feat_names):
    class_vals = [all_features[np.array(labels)==lbl, j] for lbl in [0,1,2]]
    f_stat, p_val = scipy_stats.f_oneway(*class_vals)
    print(f"    {feat:12s}: F={f_stat:.3f}, p={p_val:.4f}  {'SEPARABLE' if p_val < 0.05 else 'NOT separable (p>=0.05)'}")

# ============================================================
# SECTION 2: Model A (DeBERTa) — Independent Evaluation
# ============================================================
print("\n" + "="*70)
print("SECTION 2: MODEL A — DeBERTa Independent Evaluation")
print("="*70)

model = get_model()
tokenizer = get_tokenizer()

# Precompute features_tensors
stats_tensor = torch.tensor(all_features, dtype=torch.float32)

# Run inference on all corpus samples
all_deberta_probs = []
batch_size = 8
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_stats = stats_tensor[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], batch_stats)
        probs = F.softmax(logits, dim=1).numpy()
    all_deberta_probs.append(probs)

all_deberta_probs = np.vstack(all_deberta_probs)
deberta_preds = np.argmax(all_deberta_probs, axis=1)
labels_arr = np.array(labels)

print("\nDeBERTa Classification Report (ALL corpus samples):")
print(classification_report(labels_arr, deberta_preds, target_names=["HUMAN", "AI", "PARAPHRASED"]))

cm_d = confusion_matrix(labels_arr, deberta_preds)
print("DeBERTa Confusion Matrix (rows=true, cols=pred):")
print(f"{'':12s} {'HUMAN':>12s} {'AI':>12s} {'PARAPH':>12s}")
for i, row in enumerate(cm_d):
    print(f"  {label_names[i]:10s}: {row[0]:>12d} {row[1]:>12d} {row[2]:>12d}")

# ============================================================
# SECTION 3: Model B (Random Forest) — Independent Evaluation
# ============================================================
print("\n" + "="*70)
print("SECTION 3: MODEL B — Random Forest Independent Evaluation")
print("="*70)

try:
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    rf_probs = rf_model.predict_proba(all_features)
    rf_preds = np.argmax(rf_probs, axis=1)
    print("\nRandom Forest Classification Report (ALL corpus samples):")
    print(classification_report(labels_arr, rf_preds, target_names=["HUMAN", "AI", "PARAPHRASED"]))
    cm_r = confusion_matrix(labels_arr, rf_preds)
    print("RF Confusion Matrix (rows=true, cols=pred):")
    print(f"{'':12s} {'HUMAN':>12s} {'AI':>12s} {'PARAPH':>12s}")
    for i, row in enumerate(cm_r):
        print(f"  {label_names[i]:10s}: {row[0]:>12d} {row[1]:>12d} {row[2]:>12d}")

    print("\nRF Feature Importances:")
    for j, (feat, imp) in enumerate(zip(feat_names, rf_model.feature_importances_)):
        print(f"  {feat:12s}: {imp:.4f}")
except Exception as e:
    print(f"ERROR loading RF model: {e}")
    rf_probs = all_deberta_probs  # fallback

# ============================================================
# SECTION 4: Ensemble Logic Verification
# ============================================================
print("\n" + "="*70)
print("SECTION 4: ENSEMBLE LOGIC VERIFICATION")
print("="*70)

ensemble_probs = (all_deberta_probs * 0.6) + (rf_probs * 0.4)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
print("\nEnsemble Classification Report (60% DeBERTa + 40% RF, ALL corpus):")
print(classification_report(labels_arr, ensemble_preds, target_names=["HUMAN", "AI", "PARAPHRASED"]))

cm_e = confusion_matrix(labels_arr, ensemble_preds)
print("Ensemble Confusion Matrix:")
print(f"{'':12s} {'HUMAN':>12s} {'AI':>12s} {'PARAPH':>12s}")
for i, row in enumerate(cm_e):
    print(f"  {label_names[i]:10s}: {row[0]:>12d} {row[1]:>12d} {row[2]:>12d}")

# Show per-sample misclassification details for AI class (class 1)
print("\nMisclassified AI samples (true=AI, pred≠AI):")
for i, (true, pred) in enumerate(zip(labels_arr, ensemble_preds)):
    if true == 1 and pred != 1:
        feats = all_features[i]
        domain = domains[i]
        conf = float(ensemble_probs[i, pred])
        print(f"  [{domain}] pred={label_names[pred]} ({conf:.2%}) | PPL={feats[0]:.1f} Burst={feats[1]:.3f} TTR={feats[2]:.3f}")
        print(f"    Text: {texts[i][:100]}...")

# ============================================================
# SECTION 5: HEURISTIC OVERRIDE AUDIT
# ============================================================
print("\n" + "="*70)
print("SECTION 5: HEURISTIC OVERRIDE ANALYSIS")
print("="*70)

n_ppl_override = 0
n_informal_override = 0
n_lowppl_override = 0
n_no_override = 0

for i, text in enumerate(texts):
    feats = all_features[i]
    ppl = feats[0]
    true_label = labels_arr[i]

    has_lowercase_i = bool(re.search(r'\bi\b', text))
    has_informal = bool(re.search(r'\b(?:im|dont|cant|didnt|wont|youre|theyre|ive|couldnt|shouldnt|wasnt|isnt|aint|ngl|tbh|idk|lmao|lol|btw|gonna|wanna|bro|yall|dude|omg|smh|sus|af|fr|rn|kinda|sorta|dunno)\b', text, re.IGNORECASE))
    sentences_count = len([s for s in text.split('.') if s.strip()])

    if ppl > 500:
        n_ppl_override += 1
        if true_label != 0:
            print(f"  WRONG PPL override on class={label_names[true_label]}: [{domains[i]}] PPL={ppl:.1f}")
    elif has_lowercase_i or has_informal:
        if ensemble_preds[i] != 0:
            n_informal_override += 1
    elif ppl < 30.0 and sentences_count >= 2 and not (has_lowercase_i or has_informal):
        n_lowppl_override += 1
    else:
        n_no_override += 1

print(f"  PPL>500 overrides (→ HUMAN): {n_ppl_override}")
print(f"  Informal-text overrides (→ HUMAN): {n_informal_override}")
print(f"  Low-PPL overrides (→ AI or PARAPH): {n_lowppl_override}")

# ============================================================
# SECTION 6: GENERALIZATION — UNSEEN SAMPLES
# ============================================================
print("\n" + "="*70)
print("SECTION 6: GENERALIZATION ON UNSEEN SAMPLES")
print("="*70)

UNSEEN = [
    {"label": 1, "source": "GPT-4 article intro", "text": "The rapid proliferation of large language models has fundamentally altered the technological landscape, enabling unprecedented automation of complex cognitive tasks. These systems demonstrate remarkable proficiency across diverse domains, from scientific summarization to creative content generation. As adoption accelerates, researchers are increasingly focused on alignment, interpretability, and robustness."},
    {"label": 1, "source": "Gemini essay paragraph", "text": "Sustainable urban development requires the integration of green infrastructure within city planning frameworks. By incorporating parks, green roofs, and permeable pavements, municipalities can effectively manage stormwater runoff while reducing urban heat island effects. These interventions collectively improve resident quality of life and ecological resilience."},
    {"label": 1, "source": "Claude formal writing", "text": "The epistemological foundations of empirical science rest upon the principle of falsifiability, as articulated by Karl Popper. A claim possesses scientific validity only if it is theoretically possible to demonstrate its incorrectness through observation. This criterion distinguishes scientific inquiry from metaphysical speculation."},
    {"label": 0, "source": "Reddit casual post", "text": "honestly my laptop fan has been going insane for the last week and i have no idea why. tried closing chrome tabs but it still sounds like a jet engine. my roommate says just buy a new one but like, i'm a broke student, that's not happening lol"},
    {"label": 0, "source": "Personal blog", "text": "Three weeks into learning Portuguese and I can confidently say I now know how to order a coffee and apologize for things that aren't my fault. The subjunctive tense is going to be the death of me though. Seriously, why does every sentence need seventeen different verb forms."},
    {"label": 0, "source": "Stack Overflow rant", "text": "I've been wrestling with this CORS issue for two days now. The preflight request keeps returning 403 despite setting Access-Control-Allow-Origin to *. I've checked nginx config, FastAPI middleware, and I'm about ready to give up and just disable the browser security entirely for testing."},
    {"label": 2, "source": "QuillBot paraphrase", "text": "The swift expansion of AI-driven language systems has dramatically reshaped the digital landscape by enabling unprecedented automation of intricate mental tasks. These tools show impressive capability spanning multiple fields, ranging from scientific summarization to creative content production."},
    {"label": 2, "source": "Student AI-assisted essay", "text": "Climate change is honestly one of the scariest issues of our generation. Rising global temperatures are contributing to extreme weather events and higher sea levels, which threatens millions of coastal communities. International agreements like the Paris Accord are important, but individual countries need to actually follow through on their pledges."},
    {"label": 2, "source": "Grammarly-polished email", "text": "I wanted to reach out to discuss the Q3 project timeline. While we have made meaningful progress across several workstreams, a few deliverables are at risk of slipping due to resource constraints. I think a short sync this week could help us realign priorities and keep the project on track."},
]

unseen_true = [s["label"] for s in UNSEEN]
unseen_texts = [s["text"] for s in UNSEEN]

print(f"\nRunning inference on {len(UNSEEN)} unseen samples...")
unseen_results = []
for sample in UNSEEN:
    text = sample["text"]
    feats = nlp.extract_features(text)
    feats_t = torch.tensor([feats], dtype=torch.float32)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], feats_t)
        probs_dl = F.softmax(logits, dim=1).numpy()[0]
    probs_rf = rf_model.predict_proba([feats])[0]
    ens_probs = probs_dl * 0.6 + probs_rf * 0.4
    pred = int(np.argmax(ens_probs))
    conf = float(np.max(ens_probs))
    unseen_results.append(pred)
    correct = "✓" if pred == sample["label"] else "✗"
    print(f"\n  [{correct}] Source: {sample['source']}")
    print(f"       True={label_names[sample['label']]} | Pred={label_names[pred]} ({conf:.1%})")
    print(f"       PPL={feats[0]:.1f}, Burst={feats[1]:.3f}, TTR={feats[2]:.3f}")
    print(f"       DeBERTa: H={probs_dl[0]:.2%} AI={probs_dl[1]:.2%} P={probs_dl[2]:.2%}")
    print(f"       RF:      H={probs_rf[0]:.2%} AI={probs_rf[1]:.2%} P={probs_rf[2]:.2%}")

unseen_true_arr = np.array(unseen_true)
unseen_pred_arr = np.array(unseen_results)
print("\nUnseen Test Set Summary:")
print(classification_report(unseen_true_arr, unseen_pred_arr, target_names=["HUMAN", "AI", "PARAPHRASED"], zero_division=0))

print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70)
