import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from nlp_engine import NLPEngine
from model import get_model, get_tokenizer

LABELS = {
    0: "HUMAN-WRITTEN",
    1: "FULLY AI-GENERATED",
    2: "HUMAN-POLISHED / AI-PARAPHRASED"
}

def test():
    print("Initializing engine and models...")
    nlp = NLPEngine()
    qm = get_model()
    tok = get_tokenizer()
    
    try:
        with open("rf_model.pkl", "rb") as f:
            rf = pickle.load(f)
    except Exception:
        rf = None

    texts = [
        "Hello, my name is XYZ and I am a software engineer. I love building tools.",
        "The Real-Time AI Text Detection System is a high-performance application designed for text classification.",
        "i was thinking maybe we go to the store and get some stuff then head back home what do you think"
    ]

    for text in texts:
        print(f"\nTesting text: \"{text}\"")
        start = time.perf_counter()

        features_np = nlp.extract_features(text)
        features_tensor = torch.tensor([features_np], dtype=torch.float32)
        inputs = tok(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

        with torch.no_grad():
            logits = qm(inputs['input_ids'], inputs['attention_mask'], features_tensor)
            probs_dl = F.softmax(logits, dim=1).numpy()[0]

        if rf is not None:
            probs_rf = rf.predict_proba([features_np])[0]
            ensemble_probs = (probs_dl * 0.60) + (probs_rf * 0.40)
        else:
            ensemble_probs = probs_dl

        pred_idx = int(np.argmax(ensemble_probs))
        conf = float(ensemble_probs[pred_idx])

        # Heuristic calibration (same as tasks.py)
        import re
        has_lowercase_i = bool(re.search(r'\bi\b', text))
        has_informal = bool(re.search(r'\b(?:im|dont|cant|didnt|wont|youre|theyre|ive|couldnt|shouldnt|wasnt|isnt|aint|ngl|tbh|idk|lmao|lol|btw|gonna|wanna|bro|yall|dude|omg|smh|sus)\b', text, re.IGNORECASE))
        ppl = float(features_np[0])
        ttr = float(features_np[2])

        if ppl > 500.0:
            pred_idx = 0
            conf = 0.99
        elif (has_lowercase_i or has_informal) and pred_idx != 0:
            pred_idx = 0
            conf = max(conf, 0.88)
        elif ttr > 0.85 and pred_idx == 2:
            pred_idx = 0
            conf = 0.80

        latency = (time.perf_counter() - start) * 1000
        print(f"Features  : PPL={features_np[0]:.2f}, Burst={features_np[1]:.2f}, TTR={features_np[2]:.2f}")
        print(f"Prediction: {LABELS[pred_idx]} (Confidence: {conf*100:.1f}%)")
        print(f"Latency   : {latency:.2f}ms")

if __name__ == '__main__':
    test()
