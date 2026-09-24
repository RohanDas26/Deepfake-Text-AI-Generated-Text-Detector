import asyncio
import time
from main import PredictRequest
from nlp_engine import NLPEngine
from model import get_model, get_tokenizer
import torch
import torch.nn.functional as F

def test():
    print("Initializing engine and model...")
    nlp = NLPEngine()
    qm = get_model()
    tok = get_tokenizer()
    
    texts = [
        "Hello, my name is XYZ and I am a software engineer. I love building tools.",
        "The Real-Time AI Text Detection System is a high-performance application designed for text classification."
    ]
    
    for text in texts:
        print(f"\nTesting text: {text}")
        start = time.perf_counter()
        
        features_np = nlp.extract_features(text)
        features_tensor = torch.tensor([features_np], dtype=torch.float32)
        inputs = tok(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            logits = qm(inputs['input_ids'], inputs['attention_mask'], features_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        latency = (time.perf_counter() - start) * 1000
        print(f"Features: PPL={features_np[0]:.2f}, Burst={features_np[1]:.2f}, TTR={features_np[2]:.2f}")
        print(f"Prediction: {pred_idx.item()} (Confidence: {conf.item():.2f})")
        print(f"Latency: {latency:.2f}ms")

if __name__ == '__main__':
    test()
