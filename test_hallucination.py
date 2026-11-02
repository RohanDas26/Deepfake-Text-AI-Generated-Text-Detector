"""
=============================================================================
HALLUCINATION & EDGE-CASE STRESS TEST FOR TEXT DEEPFAKE DETECTOR
=============================================================================
Tests the trained dual-branch model and ensemble against unseen out-of-domain
edge cases to verify that the model does NOT hallucinate or misclassify.
=============================================================================
"""

import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from model import get_model, get_tokenizer
from nlp_engine import NLPEngine

LABELS = {
    0: "HUMAN-WRITTEN",
    1: "FULLY AI-GENERATED",
    2: "HUMAN-POLISHED / AI-PARAPHRASED"
}

TEST_CASES = [
    {
        "expected": 0,
        "category": "Casual Personal Message",
        "text": "Hey Sarah, are you guys still meeting up for tacos tonight? Let me know if you need me to grab extra guac on the way over."
    },
    {
        "expected": 1,
        "category": "Formulaic ChatGPT Essay",
        "text": "In contemporary society, the integration of artificial intelligence into educational curricula plays an indispensable role. Furthermore, it is crucial to recognize that automated learning modules facilitate individualized pedagogical outcomes across diverse academic disciplines."
    },
    {
        "expected": 0,
        "category": "Emotional Human Memoir",
        "text": "My father kept his old pocket watch wrapped in greasepaper inside his workbench drawer. The ticking was faint, like a tired heartbeat, but whenever I wound the brass crown, I could still hear his quiet whistling."
    },
    {
        "expected": 1,
        "category": "Generic AI Marketing Blurb",
        "text": "Unlock your true business potential with our next-generation cloud analytics suite. Seamlessly streamline your enterprise workflows, uncover actionable operational insights, and elevate customer engagement with cutting-edge digital intelligence."
    },
    {
        "expected": 2,
        "category": "Human-Polished AI Draft",
        "text": "Artificial intelligence is definitely shaking up enterprise software, but let's be honest: automated analytics are useless if your core data pipeline is full of garbage. You still need human engineers auditing the models."
    },
    {
        "expected": 0,
        "category": "High-PPL Technical Stack Trace",
        "text": "FATAL: Connection pool exhausted at line 204 in database/postgres.go: pq: sorry, too many clients already. Max connection limit 100 exceeded. Immediate SIGKILL initiated."
    },
    {
        "expected": 0,
        "category": "Run-On Stream of Thought (No Punctuation)",
        "text": "i was walking to the grocery store and then i saw this big golden retriever and it was barking at a squirrel so i stopped to pet him"
    },
    {
        "expected": 0,
        "category": "Heavy Internet Slang & Typos",
        "text": "yo bro wtf was that movie ending so trash ngl like who even wrote that garbage screenplay fr"
    },
    {
        "expected": 0,
        "category": "ALL CAPS Emotional Outburst",
        "text": "WHY ARE YOU ALWAYS DOING THIS TO ME I SWEAR TO GOD IT MAKES ABSOLUTELY ZERO SENSE"
    },
    {
        "expected": 0,
        "category": "Broken Non-Native English",
        "text": "yesterday me and my brother was going to market and buyed two apple for eat but it was raining very hard"
    },
    {
        "expected": 1,
        "category": "AI Text Without Punctuation",
        "text": "artificial intelligence systems optimize operational efficiency by automating complex data processing workflows across decentralized enterprise cloud infrastructure"
    }
]

def run_stress_test():
    print("\n" + "="*75)
    print("      HALLUCINATION & EDGE-CASE STRESS TEST (TEXT DETECTION)")
    print("="*75)

    nlp = NLPEngine()
    model = get_model()
    tokenizer = get_tokenizer()

    try:
        with open("rf_model.pkl", "rb") as f:
            rf_clf = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load Random Forest: {e}")
        rf_clf = None

    passed = 0
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES, 1):
        text = case["text"]
        expected_class = case["expected"]
        category = case["category"]

        start = time.perf_counter()

        # 1. Feature extraction
        features_np = nlp.extract_features(text)
        ppl, burst, ttr = float(np.exp(features_np[0])), float(features_np[1]), float(features_np[2])

        import re
        has_lowercase_i = bool(re.search(r'\bi\b', text))
        has_informal = bool(re.search(r'\b(?:im|dont|cant|didnt|wont|youre|theyre|ive|couldnt|shouldnt|wasnt|isnt|aint|ngl|tbh|idk|lmao|lol|btw|gonna|wanna|bro|yall|dude|omg|smh|sus|af|fr|rn|kinda|sorta|dunno)\b', text, re.IGNORECASE))

        # 2. DeBERTa Model Prediction
        feat_tensor = torch.tensor([features_np], dtype=torch.float32)
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"], feat_tensor)
            deberta_probs = F.softmax(logits, dim=1).numpy()[0]
            model_a_pred = int(np.argmax(deberta_probs))

        # 3. Random Forest Prediction
        if rf_clf is not None:
            rf_probs = rf_clf.predict_proba([features_np])[0]
            model_b_pred = int(np.argmax(rf_probs))
            # 4. Ensemble: 60% DeBERTa + 40% RF
            ensemble_probs = (0.60 * deberta_probs) + (0.40 * rf_probs)
        else:
            ensemble_probs = deberta_probs
            model_b_pred = model_a_pred

        raw_pred = int(np.argmax(ensemble_probs))
        raw_conf = float(ensemble_probs[raw_pred])

        # Heuristic calibration (synchronized with tasks.py)
        if ppl > 500:
            final_pred = 0
            final_conf = 0.95
            note = "(Heuristic Code/Log Bypass Triggered: PPL > 500)"
        elif (has_lowercase_i or has_informal) and raw_pred != 0:
            final_pred = 0
            final_conf = 0.88
            note = "(Informal Human Orthography Signature)"
        else:
            final_pred = raw_pred
            final_conf = raw_conf
            note = ""

        latency = (time.perf_counter() - start) * 1000
        is_correct = (final_pred == expected_class)
        if is_correct:
            passed += 1

        status_str = "PASS" if is_correct else "FAIL"
        print(f"\n[Test {i}/{total}] [{status_str}] - Category: {category}")
        print(f"  Snippet       : \"{text[:75]}...\"")
        print(f"  Features      : PPL={ppl:.2f} | Burstiness={burst:.3f} | TTR={ttr:.3f}")
        print(f"  DeBERTa (60%) : {LABELS[model_a_pred]}")
        print(f"  RF      (40%) : {LABELS[model_b_pred]}")
        print(f"  Final Decision: {LABELS[final_pred]} ({final_conf*100:.1f}% confidence) {note}")
        print(f"  Expected      : {LABELS[expected_class]}")
        print(f"  Latency       : {latency:.1f}ms")

    print("\n" + "="*75)
    print(f"STRESS TEST SUMMARY: {passed}/{total} PASSED ({passed/total*100:.1f}%)")
    print("="*75 + "\n")

    return passed == total

if __name__ == "__main__":
    run_stress_test()
