import os
import time
import torch
import torch.nn.functional as F
from celery import Celery
import mlflow
import nltk

# Download punkt
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
except Exception:
    pass

from nlp_engine import NLPEngine, extract_attributions
from model import get_model, get_tokenizer

# Celery app configuration
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

LABELS = {
    0: "HUMAN-WRITTEN",
    1: "FULLY AI-GENERATED",
    2: "HUMAN-POLISHED / AI-PARAPHRASED"
}

# Globals for lazy loading
nlp_engine = None
model = None
tokenizer = None

def load_models():
    global nlp_engine, model, tokenizer
    if nlp_engine is None:
        import logging
        logging.info("Initializing models in Celery worker...")
        nlp_engine = NLPEngine()
        model = get_model()
        tokenizer = get_tokenizer()
    return nlp_engine, model, tokenizer

@celery_app.task(name="run_inference")
def run_inference(text: str):
    start_time = time.perf_counter()
    
    nlp, mdl, tok = load_models()
    
    # NLP Engine Feature Extraction
    features_np = nlp.extract_features(text)
    features_tensor = torch.tensor([features_np], dtype=torch.float32)
    
    # Tokenize for Contextual Branch
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Inference
    with torch.no_grad():
        logits = mdl(inputs["input_ids"], inputs["attention_mask"], features_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        
    # Extract XAI Attributions
    attributions = extract_attributions(mdl, tok, text, features_tensor)
        
    pred_idx_val = pred_idx.item()
    confidence = conf.item()
    
    sentences = nltk.sent_tokenize(text)
    ttr = float(features_np[2])
    ppl = float(features_np[0])
    
    # Heuristic Calibration
    if ppl > 500.0:
        pred_idx_val = 0
        confidence = 0.99
    elif len(sentences) <= 1:
        if ttr > 0.85 and pred_idx_val == 2:
            pred_idx_val = 0
            confidence = 0.80
        elif ttr >= 0.95 and ppl < 50.0 and pred_idx_val == 1:
            pred_idx_val = 0
            confidence = 0.85
            
    label = LABELS[pred_idx_val]
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    metrics_dict = {
        "perplexity": float(features_np[0]),
        "burstiness": float(features_np[1]),
        "ttr": float(features_np[2])
    }
    
    # Log to MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("AI_Text_Detection")
    
    with mlflow.start_run():
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("perplexity", metrics_dict["perplexity"])
        mlflow.log_metric("burstiness", metrics_dict["burstiness"])
        mlflow.log_metric("ttr", metrics_dict["ttr"])
        mlflow.log_metric("confidence", confidence)
        mlflow.log_param("prediction", label)
        
    # Evidently AI Concept Drift Logging
    try:
        import json
        from datetime import datetime
        import logging
        logger = logging.getLogger(__name__)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "perplexity": metrics_dict["perplexity"],
            "burstiness": metrics_dict["burstiness"],
            "ttr": metrics_dict["ttr"],
            "prediction": label,
            "confidence": confidence
        }
        
        log_file = "inference_logs.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        logger.info("Successfully logged data for Evidently drift detection.")
    except Exception as e:
        logger.error(f"Error logging for drift detection: {e}")
        
    result = {
        "label": label,
        "confidence": confidence,
        "metrics": metrics_dict,
        "performance": {
            "latency_ms": latency_ms,
            "quantized": False
        },
        "token_attributions": attributions
    }
    
    return result

@celery_app.task(name="run_document_inference")
def run_document_inference(full_text: str):
    """
    Splits a long document into paragraphs, runs inference on each,
    and calculates an aggregate AI score.
    """
    paragraphs = [p.strip() for p in full_text.split('\n') if len(p.strip()) > 30]
    
    if not paragraphs:
        return {"error": "No valid text found in document."}
        
    results = []
    ai_count = 0
    human_count = 0
    mixed_count = 0
    
    for p in paragraphs:
        # We reuse the synchronous logic from run_inference but skip MLflow/Drift logging for each sub-paragraph
        # to avoid spamming the tracker, or we could call a helper. 
        # For simplicity, we call the run_inference task directly as a function.
        # Wait, run_inference is a celery task. We can call it directly as a normal python function by using its underlying function,
        # but it contains mlflow logging. That's fine.
        
        # Actually, let's just write a helper or do it directly.
        nlp, mdl, tok = load_models()
        features_np = nlp.extract_features(p)
        features_tensor = torch.tensor([features_np], dtype=torch.float32)
        inputs = tok(p, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            logits = mdl(inputs["input_ids"], inputs["attention_mask"], features_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        pred_idx_val = pred_idx.item()
        confidence = conf.item()
        
        # Heuristics
        ttr = float(features_np[2])
        ppl = float(features_np[0])
        sentences = nltk.sent_tokenize(p)
        
        if ppl > 500.0:
            pred_idx_val = 0
            confidence = 0.99
        elif len(sentences) <= 1:
            if ttr > 0.85 and pred_idx_val == 2:
                pred_idx_val = 0
                confidence = 0.80
            elif ttr >= 0.95 and ppl < 50.0 and pred_idx_val == 1:
                pred_idx_val = 0
                confidence = 0.85
                
        label = LABELS[pred_idx_val]
        
        if pred_idx_val == 1:
            ai_count += 1
        elif pred_idx_val == 0:
            human_count += 1
        else:
            mixed_count += 1
            
        results.append({
            "text": p,
            "label": label,
            "confidence": confidence,
            "perplexity": round(ppl, 2),
            "burstiness": round(float(features_np[1]), 2)
        })
        
    total = len(results)
    ai_percentage = (ai_count + (mixed_count * 0.5)) / total * 100
    
    return {
        "overall_ai_score": round(ai_percentage, 1),
        "total_paragraphs": total,
        "ai_paragraphs": ai_count,
        "human_paragraphs": human_count,
        "mixed_paragraphs": mixed_count,
        "paragraph_details": results
    }
