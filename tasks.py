import os
import time
import torch
import torch.nn.functional as F
from celery import Celery
import mlflow
import nltk
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
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
rf_model = None

def load_models():
    global nlp_engine, model, tokenizer, rf_model
    if nlp_engine is None:
        import logging
        import pickle
        logging.info("Initializing models in Celery worker...")
        nlp_engine = NLPEngine()
        model = get_model()
        tokenizer = get_tokenizer()
        try:
            with open("rf_model.pkl", "rb") as f:
                rf_model = pickle.load(f)
        except Exception as e:
            logging.error(f"Could not load Random Forest model: {e}")
            rf_model = None
    return nlp_engine, model, tokenizer, rf_model

@celery_app.task(name="run_inference")
def run_inference(text: str, compute_attributions: bool = True):
    start_time = time.perf_counter()
    
    nlp, mdl, tok, rf = load_models()
    
    # NLP Engine Feature Extraction
    features_np = nlp.extract_features(text)
    features_tensor = torch.tensor([features_np], dtype=torch.float32)
    
    # Model A: DeBERTa (Deep Learning)
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        logits = mdl(inputs["input_ids"], inputs["attention_mask"], features_tensor)
        probs_dl = F.softmax(logits, dim=1).numpy()[0]
        
    # Model B: Random Forest (Statistical ML)
    if rf is not None:
        probs_ml = rf.predict_proba([features_np])[0]
    else:
        probs_ml = probs_dl # Fallback
        
    # Ensemble: Weighted Average (65% DeBERTa Deep Contextual, 35% Random Forest Statistical)
    # DeBERTa models deep contextual representations; Random Forest models
    # statistical surface features (Perplexity, Burstiness, TTR, N-gram Entropy, Sentiment).
    ensemble_probs = (probs_dl * 0.65) + (probs_ml * 0.35)

    # NLP Perplexity Principle: Language models sample tokens from tight probability distributions.
    # When text exhibits very low perplexity (< 28.0), it is mathematically near-impossible
    # for it to be completely unassisted human writing (humans naturally exhibit PPL > 40-300+).
    ppl_val = float(np.exp(features_np[0]))
    if ppl_val < 28.0 and ensemble_probs[0] > 0.35:
        # Bayesian calibration: transfer false human confidence to AI / Paraphrased
        transfer = ensemble_probs[0] * 0.70
        ensemble_probs[0] -= transfer
        ensemble_probs[1] += transfer * 0.60
        ensemble_probs[2] += transfer * 0.40
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs)

    pred_idx_val = int(np.argmax(ensemble_probs))
    confidence = float(np.max(ensemble_probs))
        
    # Extract XAI Attributions (Using DL model only) — skipped in fast extension mode
    if compute_attributions:
        attributions = extract_attributions(mdl, tok, text, features_tensor)
    else:
        attributions = []
    
    # Pure mathematical classification derived from model probability distribution
    label = LABELS[pred_idx_val]
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # Per-model individual labels and confidences for UI display
    deberta_pred_idx = int(np.argmax(probs_dl))
    rf_pred_idx = int(np.argmax(probs_ml))

    # Quick linguistic diagnostics (sub-millisecond computations for NLP subject)
    try:
        sentiment = nlp.get_sentiment_score(text)
        entropy = nlp.get_ngram_entropy(text, n=2)
        pos = nlp.get_pos_distribution(text)
    except Exception:
        sentiment = 0.0
        entropy = 0.0
        pos = {"NN_ratio": 0.0, "VB_ratio": 0.0, "JJ_ratio": 0.0, "RB_ratio": 0.0}

    metrics_dict = {
        "perplexity": round(float(np.exp(features_np[0])), 2),
        "burstiness": round(float(features_np[1]), 4),
        "ttr": round(float(features_np[2]), 4),
        "deberta_conf": float(np.max(probs_dl)),
        "deberta_label": LABELS[deberta_pred_idx],          # NLP/PDNN: explicit DeBERTa class label
        "rf_conf": float(np.max(probs_ml)),
        "rf_label": LABELS[rf_pred_idx],                    # NLP: explicit Random Forest class label
        "sentiment": round(float(features_np[5]), 2),       # NLP: VADER sentiment compound score
        "entropy": round(float(features_np[3]), 2),         # NLP: Shannon bigram entropy
        "pos": pos                                         # NLP: POS distribution (NN, VB, JJ, RB)
    }
    
    # Non-blocking MLOps Telemetry & Drift Logging (runs in background thread to avoid latency)
    def _async_telemetry():
        try:
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
        except Exception:
            pass
        try:
            import json
            from datetime import datetime
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "perplexity": metrics_dict["perplexity"],
                "burstiness": metrics_dict["burstiness"],
                "ttr": metrics_dict["ttr"],
                "prediction": label,
                "confidence": confidence
            }
            with open("inference_logs.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    import threading
    threading.Thread(target=_async_telemetry, daemon=True).start()
        
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

@celery_app.task(name="analyze_chunk")
def analyze_chunk(chunk_text: str):
    """
    Worker task to analyze a single chunk of text.
    Runs both DeBERTa and Random Forest (Ensemble).
    """
    nlp, mdl, tok, rf = load_models()
    features_np = nlp.extract_features(chunk_text)
    features_tensor = torch.tensor([features_np], dtype=torch.float32)
    
    # Model A: DeBERTa (Deep Learning)
    inputs = tok(chunk_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        logits = mdl(inputs["input_ids"], inputs["attention_mask"], features_tensor)
        probs_dl = F.softmax(logits, dim=1).numpy()[0]
        
    # Model B: Random Forest (Statistical ML)
    if rf is not None:
        probs_ml = rf.predict_proba([features_np])[0]
    else:
        probs_ml = probs_dl # Fallback
        
    # Ensemble: Weighted Average (65% DeBERTa, 35% RF)
    ensemble_probs = (probs_dl * 0.65) + (probs_ml * 0.35)
    
    ppl_val = float(np.exp(features_np[0]))
    if ppl_val < 28.0 and ensemble_probs[0] > 0.35:
        transfer = ensemble_probs[0] * 0.70
        ensemble_probs[0] -= transfer
        ensemble_probs[1] += transfer * 0.60
        ensemble_probs[2] += transfer * 0.40
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs)

    pred_idx_val = int(np.argmax(ensemble_probs))
    confidence = float(np.max(ensemble_probs))
    
    ttr = float(features_np[2])
    ppl = ppl_val
            
    label = LABELS[pred_idx_val]
    
    return {
        "text": chunk_text,
        "label": label,
        "confidence": confidence,
        "perplexity": round(ppl, 2),
        "burstiness": round(float(features_np[1]), 2),
        "pred_idx": pred_idx_val,
        "deberta_conf": float(np.max(probs_dl)),
        "rf_conf": float(np.max(probs_ml))
    }

@celery_app.task(name="aggregate_document")
def aggregate_document(chunk_results):
    """
    Reducer task: Takes the results from all parallel chunk tasks
    and calculates the final aggregate report.
    """
    ai_count = 0
    human_count = 0
    mixed_count = 0
    
    for res in chunk_results:
        idx = res.pop("pred_idx", 0)
        if idx == 1:
            ai_count += 1
        elif idx == 0:
            human_count += 1
        else:
            mixed_count += 1
            
    total = len(chunk_results)
    if total == 0:
        return {"error": "No results to aggregate."}
        
    ai_percentage = (ai_count + (mixed_count * 0.5)) / total * 100
    
    return {
        "overall_ai_score": round(ai_percentage, 1),
        "total_paragraphs": total,
        "ai_paragraphs": ai_count,
        "human_paragraphs": human_count,
        "mixed_paragraphs": mixed_count,
        "paragraph_details": chunk_results
    }
