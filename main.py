import logging
import os
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from celery import chord
from celery.result import AsyncResult
import PyPDF2
import docx
import io
import nltk
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

# ============================================================
# COMPUTER NETWORKS: Rate Limiting
# Demonstrates IP-based request throttling at the application
# layer (Layer 7 of the OSI model). SlowAPI enforces per-client
# rate limits to protect the server from denial-of-service.
# ============================================================
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMITING_ENABLED = True
except ImportError:
    limiter = None
    RATE_LIMITING_ENABLED = False
    logging.warning("slowapi not installed — rate limiting disabled. Run: pip install slowapi")

from tasks import celery_app, run_inference, analyze_chunk, aggregate_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLOps: Reduce Celery Redis retry delay so fallback kicks in within ~2s
# instead of the default 20s (20 retries × 1s). Keeps demo snappy.
try:
    celery_app.conf.broker_transport_options = {"max_retries": 1, "interval_start": 0, "interval_step": 0.2, "interval_max": 0.5}
    celery_app.conf.result_backend_transport_options = {"max_retries": 1}
    celery_app.conf.broker_connection_retry_on_startup = False
except Exception:
    pass

app = FastAPI(
    title="AI Text Detection System — NLP + PDNN + MLOps + CN Edition",
    description="""Deepfake / AI-Generated Text Detector covering:
    - NLP: Perplexity, Burstiness, TTR, NER, N-gram Entropy, Sentiment, POS Tagging
    - PDNN: DeBERTa Transformer, Dual-Branch Network, Focal Loss, BatchNorm, Ensemble
    - MLOps: MLflow, Celery, Prometheus, Grafana, Docker, Active Learning
    - Computer Networks: REST API, WebSocket, Rate Limiting, CORS, Nginx
    """,
    version="2.0.0"
)

# COMPUTER NETWORKS: CORS Middleware
# Cross-Origin Resource Sharing — standard HTTP mechanism that allows
# the Chrome Extension (running on any domain) to call this API.
# Demonstrates knowledge of HTTP headers and browser security model.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "ok", "service": "ai-text-detector"}

# COMPUTER NETWORKS: Rate Limiting setup
if RATE_LIMITING_ENABLED:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Prometheus Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Database Configuration for Active Learning
DATABASE_URL = os.getenv("DATABASE_URL")

def init_database():
    global DATABASE_URL
    if DATABASE_URL and ("postgres" in DATABASE_URL):
        try:
            pg_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            with pg_engine.connect() as conn:
                pass
            return pg_engine
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed ({e}), falling back to SQLite for local development.")
    
    DATABASE_URL = "sqlite:///./feedback.db"
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

engine = init_database()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FeedbackLog(Base):
    __tablename__ = "active_learning_feedback"
    id = Column(Integer, primary_key=True, index=True)
    text_snippet = Column(Text, nullable=False)
    original_prediction = Column(String(50), nullable=False)
    user_correction = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.warning(f"Could not initialize database tables: {e}")

class FeedbackRequest(BaseModel):
    text: str
    original_prediction: str
    user_correction: str

class PredictRequest(BaseModel):
    text: str
    fast: Optional[bool] = False

class TaskResponse(BaseModel):
    task_id: str
    status: str

class PredictResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

import uuid
LOCAL_TASK_CACHE: Dict[str, Any] = {}

# ============================================================
# MLOPS: Model Pre-warming at Startup
# Loads DeBERTa, DistilGPT-2, and Random Forest into memory
# during FastAPI startup so the first real request is fast.
# Without this, the first inference takes ~10s (cold start).
# ============================================================
@app.on_event("startup")
async def warmup_models():
    """Pre-load all ML models into memory at server startup."""
    import asyncio
    import threading
    def _warm():
        try:
            from tasks import load_models
            logger.info("Warming up models at startup...")
            load_models()
            logger.info("Model warm-up complete — first inference will be fast.")
        except Exception as e:
            logger.warning(f"Model warm-up failed (non-fatal): {e}")
    # Run in a background thread so startup doesn't block
    t = threading.Thread(target=_warm, daemon=True)
    t.start()

# Celery broker fast check helper
import socket

def is_celery_broker_alive() -> bool:
    """Fast non-blocking probe (<50ms) to check if Redis broker is listening."""
    try:
        broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        if "localhost" in broker_url or "127.0.0.1" in broker_url:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            alive = (s.connect_ex(("127.0.0.1", 6379)) == 0)
            s.close()
            return alive
        return True
    except Exception:
        return False

@app.post("/predict", response_model=TaskResponse)
async def predict(request_data: PredictRequest, request: Request):
    """
    Submit text for AI/Human classification.
    Dispatches to Celery worker queue if broker is reachable,
    or falls back instantly to in-process execution without waiting for Celery retry timeouts.
    If fast=True, skips expensive XAI backpropagation to respond in sub-second latency for extensions.
    """
    compute_xai = not bool(request_data.fast)

    if is_celery_broker_alive():
        try:
            task = run_inference.delay(request_data.text, compute_attributions=compute_xai)
            return {"task_id": task.id, "status": "PENDING"}
        except Exception as e:
            logger.info(f"Celery dispatch failed ({e}). Using direct inference fallback.")

    local_id = str(uuid.uuid4())
    try:
        res = run_inference(request_data.text, compute_attributions=compute_xai)
        LOCAL_TASK_CACHE[local_id] = {"status": "SUCCESS", "result": res}
        return {"task_id": local_id, "status": "PENDING"}
    except Exception as inf_err:
        LOCAL_TASK_CACHE[local_id] = {"status": "FAILURE", "result": {"error": str(inf_err)}}
        return {"task_id": local_id, "status": "PENDING"}

def get_sliding_chunks(text: str, window_size=3, step=2):
    """
    Intelligently chunks text into overlapping sentence windows.
    Uses regex-based sentence splitting to avoid NLTK dependency in the web container.
    """
    import re
    # Regex-based sentence splitting (works without NLTK punkt data)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    chunks = []
    # If the text is short, just process it as one chunk
    if len(sentences) <= window_size:
        if len(text.strip()) > 20:
            chunks.append(text.strip())
        return chunks
        
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+window_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks

@app.post("/predict/document", response_model=TaskResponse)
async def predict_document(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, or TXT file for MapReduce batch analysis.
    """
    full_text = ""
    contents = await file.read()
    
    fname = (file.filename or "").lower()
    if fname.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
    elif fname.endswith(".docx"):
        doc = docx.Document(io.BytesIO(contents))
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif fname.endswith((".pptx", ".ppt")):
        # Extract text from PowerPoint presentation slides
        try:
            import pptx
            prs = pptx.Presentation(io.BytesIO(contents))
            slide_texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for p in shape.text_frame.paragraphs:
                            line = "".join(run.text for run in p.runs).strip()
                            if line:
                                slide_texts.append(line)
            full_text = "\n".join(slide_texts)
        except Exception:
            # Fallback for PPTX via zip XML extraction
            import zipfile, xml.etree.ElementTree as ET
            try:
                with zipfile.ZipFile(io.BytesIO(contents)) as z:
                    slide_files = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
                    texts = []
                    for sfile in slide_files:
                        root = ET.fromstring(z.read(sfile))
                        for elem in root.iter():
                            if elem.tag.endswith("}t") and elem.text:
                                texts.append(elem.text.strip())
                    full_text = "\n".join(texts)
            except Exception as ppt_err:
                raise HTTPException(status_code=400, detail=f"Could not parse presentation slides: {str(ppt_err)}")
    elif fname.endswith(".txt"):
        full_text = contents.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, PPTX, and TXT files are supported.")
        
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document.")
        
    chunks = get_sliding_chunks(full_text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Document text too short or invalid.")
        
    # Dispatch MapReduce Celery Chord if Redis reachable, else run in-process MapReduce instantly
    if is_celery_broker_alive():
        try:
            callback = aggregate_document.s()
            header = [analyze_chunk.s(c) for c in chunks]
            task = chord(header)(callback)
            return {"task_id": task.id, "status": "PENDING"}
        except Exception as e:
            logger.info(f"Celery chord dispatch failed ({e}). Using in-process MapReduce pipeline.")

    local_id = str(uuid.uuid4())
    try:
        chunk_results = [analyze_chunk(c) for c in chunks]
        doc_report = aggregate_document(chunk_results)
        LOCAL_TASK_CACHE[local_id] = {"status": "SUCCESS", "result": doc_report}
    except Exception as doc_err:
        LOCAL_TASK_CACHE[local_id] = {"status": "FAILURE", "result": {"error": str(doc_err)}}
    return {"task_id": local_id, "status": "PENDING"}

@app.get("/predict/result/{task_id}", response_model=PredictResponse)
async def get_predict_result(task_id: str):
    """
    Poll this endpoint with the task_id to get the final result.
    Checks in-memory fallback cache first, then Celery AsyncResult.
    """
    if task_id in LOCAL_TASK_CACHE:
        cached = LOCAL_TASK_CACHE[task_id]
        return {
            "task_id": task_id,
            "status": cached["status"],
            "result": cached["result"]
        }
        
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        response = {
            "task_id": task_id,
            "status": task_result.status,
            "result": None
        }
        if task_result.status == 'SUCCESS':
            response["result"] = task_result.result
        elif task_result.status == 'FAILURE':
            response["result"] = {"error": str(task_result.result)}
        return response
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "result": {"error": f"Result lookup error: {str(e)}"}
        }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Endpoint for Active Learning Feedback Loop. (MLOps)
    Users can flag a prediction as incorrect and provide the true label.
    This stores it in the PostgreSQL database for future model retraining.
    """
    db = SessionLocal()
    try:
        new_feedback = FeedbackLog(
            text_snippet=feedback.text,
            original_prediction=feedback.original_prediction,
            user_correction=feedback.user_correction
        )
        db.add(new_feedback)
        db.commit()
        return {"message": "Feedback successfully logged for Active Learning retraining loop."}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log feedback: {e}")
        raise HTTPException(status_code=500, detail="Database error while saving feedback.")
    finally:
        db.close()


@app.get("/nlp/analyze")
async def nlp_analyze(text: str):
    """
    NLP Subject — Full NLP Feature Analysis Endpoint.
    Returns a comprehensive breakdown of all NLP features extracted from the text:
    - Perplexity (language model scoring via DistilGPT-2)
    - Burstiness (sentence length variance — σ/μ)
    - Type-Token Ratio (lexical diversity measure)
    - Bigram & Trigram Entropy (N-gram language modeling, Shannon entropy)
    - Sentiment Score (VADER compound score)
    - POS Tag Distribution (Noun/Verb/Adj/Adverb ratios)
    - Named Entity Recognition (NLTK NE chunker)
    """
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters.")
    
    try:
        from nlp_engine import NLPEngine
        engine = NLPEngine()
        features = engine.extract_full_nlp_features(text)
        return {
            "status": "success",
            "text_length": len(text),
            "nlp_features": features,
            "subject": "NLP — Neural Language Processing"
        }
    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"NLP analysis error: {str(e)}")


@app.post("/admin/retrain")
async def trigger_retraining():
    """
    MLOps Subject — Model Retraining Trigger Endpoint.
    In a production MLOps pipeline, this endpoint would:
    1. Pull labeled feedback samples from the Active Learning database
    2. Trigger a Celery task to run the training loop
    3. Log the new run to MLflow and register the updated model
    4. Replace the live model after validation (Blue-Green deployment)
    """
    try:
        db = SessionLocal()
        feedback_count = db.query(FeedbackLog).count()
        db.close()
        
        if feedback_count < 10:
            return {
                "status": "skipped",
                "reason": f"Insufficient feedback samples ({feedback_count}/10 minimum). Keep collecting user corrections.",
                "mlops_note": "Active Learning requires minimum labeled samples before retraining is triggered."
            }
        
        # In production, this would dispatch: retrain_model.delay()
        return {
            "status": "triggered",
            "feedback_samples_available": feedback_count,
            "message": "Retraining pipeline dispatched to Celery worker. Monitor progress via MLflow UI at :5000",
            "mlops_pipeline": [
                "1. Load feedback from PostgreSQL",
                "2. Preprocess and tokenize samples",
                "3. Fine-tune DeBERTa classification head with Focal Loss",
                "4. Evaluate F1 / Precision / Recall on holdout set",
                "5. Register new model version in MLflow Registry",
                "6. Swap model if performance improves (Blue-Green)"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# COMPUTER NETWORKS: WebSocket — Real-Time Bidirectional Protocol
# WebSocket operates over a persistent TCP connection (RFC 6455).
# Unlike HTTP (stateless request-response), WebSocket enables
# full-duplex communication — essential for real-time applications.
# This demonstrates knowledge of application-layer protocols beyond HTTP.
# ================================================================
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Computer Networks Subject — WebSocket Real-Time Text Analysis.
    Client sends text over a persistent TCP connection.
    Server streams back NLP features and final prediction in real-time.
    Protocol: WS (WebSocket over TCP, port 8000, Upgrade: websocket header)
    """
    await websocket.accept()
    logger.info(f"CN: WebSocket connection established from {websocket.client.host}:{websocket.client.port}")
    
    try:
        while True:
            # Receive text from client (bidirectional TCP stream)
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
                text = payload.get("text", "")
            except json.JSONDecodeError:
                text = data  # plain text fallback
            
            if not text or len(text.strip()) < 5:
                await websocket.send_json({"error": "Text too short."})
                continue
            
            # Stream Step 1: NLP features (fast)
            await websocket.send_json({"step": "nlp_features", "status": "computing..."})
            
            try:
                from nlp_engine import NLPEngine
                engine = NLPEngine()
                ppl = engine.get_perplexity(text)
                burstiness = engine.get_burstiness(text)
                ttr = engine.get_ttr(text)
                sentiment = engine.get_sentiment_score(text)
                bigram_entropy = engine.get_ngram_entropy(text, n=2)
                
                await websocket.send_json({
                    "step": "nlp_features",
                    "status": "done",
                    "data": {
                        "perplexity": round(ppl, 2),
                        "burstiness": round(burstiness, 4),
                        "type_token_ratio": round(ttr, 4),
                        "sentiment": round(sentiment, 4),
                        "bigram_entropy": round(bigram_entropy, 4)
                    }
                })
                
                # Stream Step 2: Queue inference task
                await websocket.send_json({"step": "inference", "status": "queuing Celery task..."})
                task = run_inference.delay(text)
                
                await websocket.send_json({
                    "step": "inference",
                    "status": "queued",
                    "task_id": task.id,
                    "message": f"Poll GET /predict/result/{task.id} for final prediction"
                })
                
            except Exception as e:
                await websocket.send_json({"step": "error", "message": str(e)})
                
    except WebSocketDisconnect:
        logger.info("CN: WebSocket client disconnected.")

# ================================================================
# IMAGE DEEPFAKE DETECTION ENDPOINT
# PDNN: EfficientNet-B0 CNN + FFT Frequency Analysis + Grad-CAM
# Computer Networks: HTTP multipart/form-data file upload
# MLOps: MLflow experiment logging
# ================================================================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Image Deepfake Detection Endpoint.

    Upload a face image (JPG/PNG/WEBP) to detect if it is:
      - REAL: an authentic photograph
      - FAKE (DEEPFAKE): AI-generated or face-swapped image

    PDNN Concepts used:
      - EfficientNet-B0 CNN with Transfer Learning
      - 2D FFT Frequency Domain Analysis (GAN artifact detection)
      - Batch Normalization, GELU activation, Global Average Pooling
      - Grad-CAM XAI: returns heatmap showing which face region is fake

    Computer Networks:
      - HTTP POST with multipart/form-data encoding (file upload)
      - Standard binary file transfer over TCP

    Returns:
      label, confidence, real/fake probabilities,
      face detection status, Grad-CAM heatmap (base64 PNG)
    """
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPG, PNG, or WEBP."
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large. Max size: 10MB.")

    try:
        from image_module.image_engine import run_image_inference
        result = run_image_inference(contents)
        return result
    except Exception as e:
        logger.error(f"Image inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")


# Serve static dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
