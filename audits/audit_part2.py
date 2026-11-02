# -*- coding: utf-8 -*-
"""
Audit Part 2 — Heuristic analysis + Unseen generalization test
(Run after audit_pipeline.py completes sections 0-4)
"""
import os, sys, pickle, re, logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

from data_corpus import CORPUS
from model import get_model, get_tokenizer
from nlp_engine import NLPEngine

label_names = {0: "HUMAN", 1: "AI", 2: "PARAPHRASED"}
texts = [c["text"] for c in CORPUS]
labels = [c["label"] for c in CORPUS]
domains = [c["domain"] for c in CORPUS]
labels_arr = np.array(labels)

# Load models
nlp = NLPEngine()
model = get_model()
tokenizer = get_tokenizer()
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Compute features for full corpus
print("Computing features for all corpus samples...")
all_features = np.array([nlp.extract_features(t) for t in texts])
stats_tensor = torch.tensor(all_features, dtype=torch.float32)

# Run DeBERTa
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

rf_probs = rf_model.predict_proba(all_features)
ensemble_probs = (all_deberta_probs * 0.6) + (rf_probs * 0.4)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# ============================================================
# SECTION 5: HEURISTIC OVERRIDE AUDIT
# ============================================================
print("\n" + "="*70)
print("SECTION 5: HEURISTIC OVERRIDE ANALYSIS")
print("="*70)

for i, text in enumerate(texts):
    feats = all_features[i]
    ppl = feats[0]
    true_label = labels_arr[i]

    has_lowercase_i = bool(re.search(r'\bi\b', text))
    has_informal = bool(re.search(
        r'\b(?:im|dont|cant|didnt|wont|youre|theyre|ive|couldnt|shouldnt|wasnt|isnt|aint|ngl|tbh|idk|lmao|lol|btw|gonna|wanna|bro|yall|dude|omg|smh|sus|af|fr|rn|kinda|sorta|dunno)\b',
        text, re.IGNORECASE))
    sentences_count = len([s for s in text.split('.') if s.strip()])

    if ppl > 500:
        if true_label != 0:
            print(f"  [WRONG PPL>500 override] True={label_names[true_label]}: [{domains[i]}] PPL={ppl:.1f}")
            print(f"    Text: {text[:100]}")
    elif (has_lowercase_i or has_informal) and ensemble_preds[i] != 0:
        print(f"  [Informal override needed] True={label_names[true_label]} Pred={label_names[ensemble_preds[i]]}: [{domains[i]}] PPL={ppl:.1f}")
    elif ppl < 30.0 and sentences_count >= 2 and not (has_lowercase_i or has_informal):
        if true_label == 0:
            print(f"  [WRONG low-PPL override on HUMAN] [{domains[i]}] PPL={ppl:.1f}")

print("\nMisclassified AI samples (true=AI, pred!=AI):")
any_found = False
for i, (true, pred) in enumerate(zip(labels_arr, ensemble_preds)):
    if true == 1 and pred != 1:
        any_found = True
        feats = all_features[i]
        domain = domains[i]
        conf = float(ensemble_probs[i, pred])
        print(f"  [{domain}] pred={label_names[pred]} ({conf:.2%}) | PPL={feats[0]:.1f} Burst={feats[1]:.3f} TTR={feats[2]:.3f}")
        print(f"    Text: {texts[i][:120]}")
        print(f"    DeBERTa: H={all_deberta_probs[i,0]:.2%} AI={all_deberta_probs[i,1]:.2%} P={all_deberta_probs[i,2]:.2%}")
        print(f"    RF:      H={rf_probs[i,0]:.2%} AI={rf_probs[i,1]:.2%} P={rf_probs[i,2]:.2%}")
if not any_found:
    print("  None - all 34 AI samples correctly classified by ensemble.")

# ============================================================
# SECTION 6: GENERALIZATION TEST ON UNSEEN SAMPLES
# ============================================================
print("\n" + "="*70)
print("SECTION 6: GENERALIZATION ON UNSEEN SAMPLES (not in CORPUS)")
print("="*70)

UNSEEN = [
    {"label": 1, "source": "GPT-4 article intro",
     "text": "The rapid proliferation of large language models has fundamentally altered the technological landscape, enabling unprecedented automation of complex cognitive tasks. These systems demonstrate remarkable proficiency across diverse domains, from scientific summarization to creative content generation. As adoption accelerates, researchers are increasingly focused on alignment, interpretability, and robustness."},
    {"label": 1, "source": "Gemini essay paragraph",
     "text": "Sustainable urban development requires the integration of green infrastructure within city planning frameworks. By incorporating parks, green roofs, and permeable pavements, municipalities can effectively manage stormwater runoff while reducing urban heat island effects. These interventions collectively improve resident quality of life and ecological resilience."},
    {"label": 1, "source": "Claude formal writing",
     "text": "The epistemological foundations of empirical science rest upon the principle of falsifiability, as articulated by Karl Popper. A claim possesses scientific validity only if it is theoretically possible to demonstrate its incorrectness through observation. This criterion distinguishes scientific inquiry from metaphysical speculation."},
    {"label": 1, "source": "ChatGPT casual helpful reply",
     "text": "Of course! Here are some effective study strategies for your upcoming exam. First, use active recall techniques by testing yourself regularly rather than passively rereading notes. Second, space out your study sessions over multiple days using the spaced repetition method. Finally, prioritize understanding core concepts over memorizing specific details."},
    {"label": 0, "source": "Reddit casual post",
     "text": "honestly my laptop fan has been going insane for the last week and i have no idea why. tried closing chrome tabs but it still sounds like a jet engine. my roommate says just buy a new one but like, i'm a broke student, that's not happening lol"},
    {"label": 0, "source": "Personal blog",
     "text": "Three weeks into learning Portuguese and I can confidently say I now know how to order a coffee and apologize for things that aren't my fault. The subjunctive tense is going to be the death of me though. Seriously, why does every sentence need seventeen different verb forms."},
    {"label": 0, "source": "Stack Overflow rant",
     "text": "I've been wrestling with this CORS issue for two days now. The preflight request keeps returning 403 despite setting Access-Control-Allow-Origin to *. I've checked nginx config, FastAPI middleware, and I'm about ready to give up and just disable the browser security entirely for testing."},
    {"label": 0, "source": "WhatsApp message style",
     "text": "bro did you see that match yesterday?? insane ending honestly didn't expect them to come back from 3-0 down like that. my whole family was freaking out haha"},
    {"label": 2, "source": "QuillBot paraphrase of GPT-4",
     "text": "The swift expansion of AI-driven language systems has dramatically reshaped the digital landscape by enabling unprecedented automation of intricate mental tasks. These tools show impressive capability spanning multiple fields, ranging from scientific summarization to creative content production."},
    {"label": 2, "source": "Student AI-assisted essay",
     "text": "Climate change is honestly one of the scariest issues of our generation. Rising global temperatures are contributing to extreme weather events and higher sea levels, which threatens millions of coastal communities. International agreements like the Paris Accord are important, but individual countries need to actually follow through on their pledges."},
    {"label": 2, "source": "Grammarly-polished email",
     "text": "I wanted to reach out to discuss the Q3 project timeline. While we have made meaningful progress across several workstreams, a few deliverables are at risk of slipping due to resource constraints. I think a short sync this week could help us realign priorities and keep the project on track."},
    {"label": 2, "source": "ChatGPT result lightly edited by human",
     "text": "You know, sometimes I feel like one of the most interesting phases of life is the one where you know you want something big, but haven't quite figured out how to get there yet. You're building skills, trying new things, making plans. Here's something worth keeping in mind: you don't necessarily need to have your future figured out right now."},
]

unseen_true = [s["label"] for s in UNSEEN]
unseen_preds = []

print(f"\nRunning inference on {len(UNSEEN)} unseen samples...\n")
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
    unseen_preds.append(pred)
    correct = "OK" if pred == sample["label"] else "WRONG"
    print(f"  [{correct}] Source: {sample['source']}")
    print(f"       True={label_names[sample['label']]:12s} | Pred={label_names[pred]:12s} ({conf:.1%})")
    print(f"       PPL={feats[0]:.1f}, Burst={feats[1]:.3f}, TTR={feats[2]:.3f}")
    print(f"       DeBERTa: H={probs_dl[0]:.2%} AI={probs_dl[1]:.2%} P={probs_dl[2]:.2%}")
    print(f"       RF:      H={probs_rf[0]:.2%} AI={probs_rf[1]:.2%} P={probs_rf[2]:.2%}")
    print()

unseen_true_arr = np.array(unseen_true)
unseen_pred_arr = np.array(unseen_preds)
print("Unseen Test Set Summary:")
print(classification_report(unseen_true_arr, unseen_pred_arr, target_names=["HUMAN", "AI", "PARAPHRASED"], zero_division=0))

ai_correct = sum(1 for t, p in zip(unseen_true, unseen_preds) if t==1 and p==1)
ai_total = sum(1 for t in unseen_true if t==1)
human_correct = sum(1 for t, p in zip(unseen_true, unseen_preds) if t==0 and p==0)
human_total = sum(1 for t in unseen_true if t==0)
para_correct = sum(1 for t, p in zip(unseen_true, unseen_preds) if t==2 and p==2)
para_total = sum(1 for t in unseen_true if t==2)

print(f"Per-class accuracy on unseen:")
print(f"  HUMAN-WRITTEN:        {human_correct}/{human_total} = {human_correct/human_total:.0%}")
print(f"  FULLY AI-GENERATED:   {ai_correct}/{ai_total} = {ai_correct/ai_total:.0%}")
print(f"  AI-PARAPHRASED:       {para_correct}/{para_total} = {para_correct/para_total:.0%}")

print("\n" + "="*70)
print("AUDIT PART 2 COMPLETE")
print("="*70)
