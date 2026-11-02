# 🧠 AGENT MEMORY — AI Text & Image Deepfake Detection System
> **READ THIS FIRST before making any changes to this project.**
> This file is the single source of truth for any AI agent or developer working on this codebase.

---

## 📌 Project Identity

| Field | Value |
|---|---|
| **Project Name** | Multi-Modal Deepfake Detection System |
| **Owner** | Rohan Das |
| **Type** | 3rd Year B.Tech Final Project |
| **GitHub Repo** | RohanDas26/Deepfake-Text-AI-Generated-Text-Detector |
| **Local Path** | `c:\Users\Rohan Das\Desktop\NLP_PDNC_MLOPS_PROJECT` |

### Academic Purpose
This project is the **combined implementation** for **3 core university subjects**:
1. **NLP** — Neural Language Processing
2. **PDNN** — Principles of Deep Neural Networks (PDNC)
3. **MLOps** — Machine Learning Operations

*(Note: Computer Networks (CN) theoretical modules have been archived into `audits/` to keep the presentation and implementation focused, clean, and straightforward.)*

Every feature in this project must map to at least one of these subjects.
When adding new features, always tag them with the relevant subject.

---

## 🎯 What This Project Does

A **Multi-Modal AI Deepfake Detector** that can detect:
- **Text**: Whether text is AI-generated, human-written, or AI-paraphrased (3 classes)
- **Image**: Whether a face image is real or a deepfake (2 classes)

The system is production-grade with a full MLOps stack, REST API, real-time WebSocket, Chrome Extension client, and Docker containerization.

---

## 📁 Complete File Map

### Core Text ML Files (Primary System)
| File | Purpose | Subject |
|---|---|---|
| `model.py` | `PDNCDualBranchNetwork` — DeBERTa v3 + Statistical branch fusion model | PDNN |
| `nlp_engine.py` | `NLPEngine` — Feature extractor: perplexity, burstiness, TTR, NER, n-gram entropy, sentiment, POS | NLP |
| `data_corpus.py` | 90-sample multi-domain benchmark dataset across all 3 classes | NLP + PDNN |
| `tasks.py` | Celery async task definitions: `run_inference`, `analyze_chunk`, `aggregate_document` | MLOps |
| `train.py` | Deep text model training: Focal Loss, AdamW, CosineAnnealingLR, validation checkpointing | PDNN + MLOps |
| `train_rf.py` | Random Forest ensemble model training on real NLP features | NLP + MLOps |
| `test_inference.py` | Standalone verification script for testing inference, feature extraction, and latency | PDNN + MLOps |
| `test_hallucination.py` | Hand-crafted 11-case sanity check script (60/40 ensemble + heuristic regex overrides) | Diagnostic |

### API & Server (Text-Focused)
| File | Purpose | Subject |
|---|---|---|
| `main.py` | FastAPI app: all REST endpoints, WebSocket, rate limiting, CORS | CN + MLOps + NLP |
| `network_demo.py` | Standalone CN demo: TCP sockets, DNS, HTTP, OSI model port mapping | CN |

### Isolated Secondary Module (`image_module/`)
| File | Purpose | Subject |
|---|---|---|
| `image_module/image_engine.py` | `ImageDeepfakeDetector` — EfficientNet-B0 CNN + FFT frequency analysis + Grad-CAM | PDNN |
| `image_module/train_image.py` | Image model training: EfficientNet-B0, data augmentation, Focal Loss, MLflow Registry | PDNN + MLOps |
| `image_module/README.md` | Module documentation & isolation policy | — |

### Frontend & Static Assets
| File | Description |
|---|---|
| `static/index.html` | Interactive web dashboard UI for text and image deepfake analysis |

### Infrastructure / MLOps
| File | Purpose | Subject |
|---|---|---|
| `docker-compose.yml` | Orchestrates: web, worker, redis, postgres, mlflow, prometheus, grafana, nginx | MLOps |
| `Dockerfile` | Multi-stage Docker build for the FastAPI app | MLOps |
| `nginx.conf` | Nginx reverse proxy: load balancing, WebSocket upgrade, rate limiting, X-Forwarded-For | CN |
| `requirements.txt` | All Python dependencies | — |
| `prometheus/prometheus.yml` | Prometheus scrape configuration | MLOps |
| `.github/workflows/ci_cd.yml` | GitHub Actions CI/CD pipeline: test → build → push Docker → drift check | MLOps |

### Models (Binary)
| File | Description |
|---|---|
| `model.pt` | Trained `PDNCDualBranchNetwork` weights (~283MB, DeBERTa v3 small) |
| `rf_model.pkl` | Trained Random Forest model (~107KB, scikit-learn) |
| `image_model.pt` | Trained `ImageDeepfakeDetector` weights (created after running `train_image.py`) |

### Data & Logs
| File | Description |
|---|---|
| `inference_logs.jsonl` | JSONL log of all text inference requests (used for drift detection) |
| `mlflow.db` | SQLite MLflow tracking database (local dev fallback) |
| `feedback.db` | SQLite local database fallback for Active Learning feedback when PostgreSQL is offline |
| `test_upload.txt` | Sample text file for testing document upload endpoint |

### Chrome Extension
| File | Description |
|---|---|
| `chrome_extension/manifest.json` | Chrome Extension manifest |
| `chrome_extension/background.js` | Extension background script — sends selected text to API |

---

## 🏗️ System Architecture

```
CLIENT LAYER
─────────────────────────────────────────────────────────────────
  Browser / Chrome Extension / WebSocket Client
         |
         |  HTTP (port 80)  /  WebSocket (ws://)
         v
NETWORK LAYER (Computer Networks Subject)
─────────────────────────────────────────────────────────────────
  Nginx Reverse Proxy (nginx.conf)
    - Layer 7 load balancing (Round Robin)
    - WebSocket protocol upgrade (RFC 6455)
    - Rate limiting (token bucket, 30 req/min)
    - X-Real-IP / X-Forwarded-For header forwarding
         |
         |  proxy_pass to port 8000
         v
APPLICATION LAYER (CN + MLOps Subject)
─────────────────────────────────────────────────────────────────
  FastAPI Web Server (main.py, port 8000)
    REST Endpoints:
      POST /predict              -> async text inference (Celery task)
      POST /predict/document     -> PDF/DOCX/TXT MapReduce analysis
      GET  /predict/result/{id}  -> poll for task result
      POST /predict/image        -> image deepfake detection
      GET  /nlp/analyze          -> full NLP feature breakdown
      POST /feedback             -> active learning feedback log
      POST /admin/retrain        -> trigger model retraining
      GET  /metrics              -> Prometheus scrape endpoint
      WS   /ws/predict           -> real-time WebSocket streaming
         |
         +- Dispatches Celery tasks -> Redis (port 6379)
         +- Writes feedback -> PostgreSQL (port 5432) [fallback: feedback.db (SQLite)]

WORKER LAYER (MLOps Subject)
─────────────────────────────────────────────────────────────────
  Celery Worker (tasks.py)
    - Consumes tasks from Redis queue
    - Runs text inference (DeBERTa + RF ensemble)
    - Logs metrics to MLflow
    - Writes drift data to inference_logs.jsonl

ML LAYER (PDNN + NLP Subject)
─────────────────────────────────────────────────────────────────
  Text Pipeline:
    NLPEngine (nlp_engine.py)
      -> Perplexity (DistilGPT-2)
      -> Burstiness (sigma/mu sentence lengths)
      -> TTR (type-token ratio)
      -> N-gram entropy (Shannon H, bigram+trigram)
      -> Sentiment (VADER compound score)
      -> POS distribution (NLTK pos_tag)
      -> NER entities (NLTK ne_chunk)
      |
    PDNCDualBranchNetwork (model.py)
      Branch A: DeBERTa v3 small -> mean pooling -> 768-dim
      Branch B: Statistical [3-dim] -> Linear -> BatchNorm1d -> GELU -> 32-dim
      Fusion: Concat(768+32) -> Linear -> BatchNorm1d -> Dropout -> GELU -> 3 classes
      |
    Random Forest (rf_model.pkl) -- on same NLP features
      |
    Ensemble: 60% DeBERTa + 40% RF -> final prediction
      |
    XAI: Input x Gradient saliency (token-level attributions)

  Image Pipeline:
    Face Detection (OpenCV Haar Cascade)
      |
    ImageDeepfakeDetector (image_engine.py)
      Branch A: EfficientNet-B0 CNN (Transfer Learning, frozen backbone) -> 1280-dim
      Branch B: 2D FFT -> log magnitude -> 16x16 -> Linear -> BatchNorm1d -> GELU -> 64-dim
      Fusion: Concat(1280+64) -> Linear -> BatchNorm1d -> GELU -> Dropout -> 2 classes
      |
    XAI: Grad-CAM heatmap (gradient x activation at last conv layer)

OBSERVABILITY LAYER (MLOps Subject)
─────────────────────────────────────────────────────────────────
  MLflow (port 5000)       -> experiment tracking, model registry
  Prometheus (port 9090)   -> time-series metrics scraping
  Grafana (port 3000)      -> metrics visualization dashboards
  inference_logs.jsonl     -> drift detection data store
```

---

## 🔌 All API Endpoints

| Method | Endpoint | What it does | Subject |
|---|---|---|---|
| `POST` | `/predict` | Submit text -> get `task_id` | NLP + MLOps |
| `GET` | `/predict/result/{task_id}` | Poll for text prediction result | CN |
| `POST` | `/predict/document` | Upload PDF/DOCX/TXT for MapReduce analysis | NLP + MLOps |
| `POST` | `/predict/image` | Upload face image -> REAL/FAKE + Grad-CAM | PDNN + CN |
| `GET` | `/nlp/analyze?text=...` | Full NLP feature extraction breakdown | NLP |
| `POST` | `/feedback` | Submit label correction (Active Learning) | MLOps |
| `POST` | `/admin/retrain` | Trigger model retraining pipeline | MLOps |
| `GET` | `/metrics` | Prometheus metrics scrape | MLOps + CN |
| `WS` | `/ws/predict` | Real-time WebSocket streaming analysis | CN |
| `GET` | `/` | Serve web dashboard (`static/index.html`) | — |
| `GET` | `/docs` | FastAPI auto-generated Swagger UI | — |

---

## 🧪 Subject -> Implementation Mapping

### 🧠 NLP (Neural Language Processing)
- **Perplexity**: DistilGPT-2 cross-entropy loss as language model score
- **Burstiness**: sigma/mu of sentence lengths (coefficient of variation)
- **Type-Token Ratio (TTR)**: unique_words / total_words — lexical diversity
- **N-gram Entropy**: Shannon entropy H = -sum(p(ngram) log2(p(ngram))) for bigrams + trigrams
- **Sentiment Analysis**: VADER SentimentIntensityAnalyzer — compound score in [-1, 1]
- **POS Tagging**: NLTK pos_tag — noun/verb/adjective/adverb ratios
- **Named Entity Recognition**: NLTK ne_chunk — entity count, density, type diversity
- **XAI Token Attribution**: Input x Gradient saliency on DeBERTa embeddings
- **Sliding Window Chunking**: overlapping sentence windows for document analysis
- **MapReduce**: Celery chord — parallel chunk analysis (map) + aggregation (reduce)

### 🏗️ PDNN (Principles of Deep Neural Networks)
- **Transformer Architecture**: DeBERTa v3 small (disentangled attention + enhanced mask decoder)
- **CNN Architecture**: EfficientNet-B0 (compound scaling — depth x width x resolution)
- **Transfer Learning**: Pretrained ImageNet weights, frozen backbone, trainable classification head
- **Dual-Branch Fusion**: Multi-modal feature concatenation (contextual + statistical / CNN + FFT)
- **Mean Pooling**: Attention-weighted average of token embeddings
- **Global Average Pooling**: Spatial feature reduction in CNN
- **Batch Normalization**: `nn.BatchNorm1d` — reduces internal covariate shift
- **Activation Functions**: `ReLU`, `GELU` (Gaussian Error Linear Unit)
- **Dropout Regularization**: 0.2-0.4 dropout rates in fusion and classifier heads
- **Focal Loss**: FL = (1-pt)^gamma * CE, gamma=2.0 — focuses on hard examples
- **AdamW Optimizer**: Adaptive LR + decoupled L2 weight decay
- **CosineAnnealingLR**: eta_t = eta_min + 0.5*(eta_max-eta_min)*(1+cos(pi*t/T))
- **Gradient Clipping**: `clip_grad_norm_(max_norm=1.0)` — prevents exploding gradients
- **Data Augmentation**: RandomCrop, HorizontalFlip, ColorJitter, RandomRotation
- **Ensemble Learning**: Weighted average (60% DeBERTa + 40% Random Forest)
- **Grad-CAM XAI**: Gradient x activation at last conv layer -> heatmap
- **2D FFT**: Frequency domain analysis for GAN artifact detection
- **Evaluation**: F1, Precision, Recall, Confusion Matrix, ROC-AUC (all logged to MLflow)

### ⚙️ MLOps (Machine Learning Operations)
- **Experiment Tracking**: MLflow — logs metrics, params, artifacts per run
- **Model Registry**: MLflow registry — `AITextDetector`, `ImageDeepfakeDetector`
- **Async Task Queue**: Celery + Redis — decouples inference from API serving
- **MapReduce Pipeline**: Celery `chord` — parallel document analysis
- **Containerization**: Multi-stage Dockerfile (builder + runtime stages)
- **Orchestration**: Docker Compose — 8 services (web, worker, redis, postgres, mlflow, prometheus, grafana, nginx)
- **Monitoring**: Prometheus + Grafana — latency, confidence, throughput dashboards
- **Active Learning**: PostgreSQL feedback table — collects user label corrections
- **Concept Drift**: JSONL inference log -> statistical drift detection (perplexity distribution shift)
- **CI/CD**: GitHub Actions — test -> validate architecture -> build Docker -> push to Docker Hub -> drift check
- **Retraining Trigger**: `POST /admin/retrain` — checks feedback count, dispatches retrain task
- **Blue-Green Deployment**: Documented in retrain endpoint response

### 🌐 Computer Networks
- **REST API**: HTTP/1.1 request-response over TCP (FastAPI)
- **WebSocket**: Full-duplex TCP streaming (RFC 6455) — `ws://host/ws/predict`
- **CORS**: Cross-Origin Resource Sharing headers — allows Chrome Extension on any domain
- **Rate Limiting**: SlowAPI (Layer 7) + Nginx `limit_req` (Layer 7) — 30 req/min per IP
- **Reverse Proxy**: Nginx — hides backend, handles routing, WebSocket upgrade
- **Load Balancing**: Nginx upstream Round-Robin pool
- **HTTP Headers**: X-Real-IP, X-Forwarded-For, Upgrade, Connection
- **TCP Sockets**: Raw socket demo in `network_demo.py`
- **DNS Resolution**: Docker Compose internal DNS (127.0.0.11 resolver)
- **OSI Model**: Full layer mapping documented in `network_demo.py`
- **Message Broker**: Redis RESP protocol over TCP (port 6379)
- **Chrome Extension**: Client-server architecture — extension <-> API
- **Multipart Upload**: HTTP `multipart/form-data` for PDF, DOCX, image uploads
- **Prometheus Pull**: HTTP GET scrape from `/metrics` every 15s

---

## 🚀 How to Run

### Development (Fast — No Docker)
```powershell
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Celery worker (add '-P solo' on Windows to avoid prefork concurrency issues)
celery -A tasks worker --loglevel=info -P solo

# Open Swagger UI: http://localhost:8000/docs
```

### Full MLOps Stack (Docker — For Demo/Exam Only)
```powershell
docker-compose up --build
# Takes 2-3 mins. Services:
# http://localhost      -> FastAPI via Nginx (port 80)
# http://localhost:8000 -> FastAPI direct
# http://localhost:5000 -> MLflow UI
# http://localhost:9090 -> Prometheus
# http://localhost:3000 -> Grafana
```

---

## ⚠️ Rules for Future Agents

### DO NOT change:
- The 3-class label structure: `HUMAN-WRITTEN` / `FULLY AI-GENERATED` / `HUMAN-POLISHED / AI-PARAPHRASED`
- The `model.pt` weights unless explicitly asked to retrain
- The `mlflow.db` SQLite file — it has existing experiment history
- The Celery task names (`run_inference`, `analyze_chunk`, `aggregate_document`) — Chrome Extension depends on these
- **NEVER merge or couple text and image code/logic**: Text is the **PRIMARY** project. Image deepfake detection must always remain an isolated, decoupled secondary module (`image_engine.py` and `train_image.py`). Do not blend their data structures, pipelines, or models.

### ALWAYS:
- Tag new features with their relevant subject (`NLP`, `PDNN`, `MLOps`, `CN`) in comments
- Log new inference metrics to MLflow
- Maintain backward compatibility with `/predict` endpoint (Chrome Extension depends on it)
- Add new Python dependencies to `requirements.txt`

### Architecture Constraints:
- Text & Image isolation: Text is the primary core. All text files (`model.py`, `nlp_engine.py`, `tasks.py`, `train.py`, `train_rf.py`) must remain 100% standalone with zero image dependencies. Image logic is strictly confined to `image_engine.py` and `train_image.py`.
- Text model input: 3-dim statistical feature vector `[perplexity, burstiness, ttr]`
- DeBERTa max token length: 512 tokens (hard limit)
- Image model input: 224x224x3 normalized with ImageNet mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`
- Celery broker: Redis at `CELERY_BROKER_URL` env var (default: `redis://localhost:6379/0`)
- MLflow URI: `MLFLOW_TRACKING_URI` env var (default: `sqlite:///mlflow.db`)

---

## 📊 Trained Models Info

### `model.pt` — PDNCDualBranchNetwork
- **Backbone**: `microsoft/deberta-v3-small` (frozen feature extractor with parameters frozen at `requires_grad=False`; only the statistical projection branch and fusion MLP head are trained).
- **Trained on**: 555 balanced multi-domain benchmark samples (185 Human, 185 AI, 185 Paraphrased) combining 195 local multi-domain corpus samples + 360 curated samples from the RAID benchmark (curated 360-sample slice across 5 domains, not the full 14GB dataset).
- **Features**: Deep contextual token embeddings (768-dim) from frozen DeBERTa + 8-dim statistical NLP features (`[log_ppl, burstiness, ttr, bigram_entropy, trigram_entropy, sentiment, nn_ratio, vb_ratio]`) projected to 32-dim.
- **Loss**: Focal Loss (gamma=2.0)
- **Optimizer**: AdamW (lr=8e-4, weight_decay=1e-4) + CosineAnnealingLR (70 epochs)
- **Individual DeBERTa Validation**: 90.99% Accuracy, 90.88% Macro F1 on the 111-sample validation split.
- **Classes**:
  - `Class 0 (HUMAN-WRITTEN)`: Unmodified human text.
  - `Class 1 (FULLY AI-GENERATED)`: Machine-generated text with no adversarial rewrite attack.
  - `Class 2 (HUMAN-POLISHED / AI-PARAPHRASED)`: Text altered by RAID paraphrase or synonym perturbation attacks (original source text may be human or AI).

### `rf_model.pkl` — Random Forest
- **Features**: 8-dim statistical NLP features extracted from 444 training samples.
- **Configuration**: 100 estimators, max depth 6.
- **Individual RF Validation**: 72.07% Accuracy on the 111-sample validation split.

### System Performance on In-Distribution Validation Set (111 samples)
> *Note on Validation Optimism*: The 111-sample set functioned as a validation and tuning set (used for checkpoint selection across 70 epochs, manual selection of the 65/35 ensemble ratio, and threshold tuning). Therefore, the **87.39% accuracy** is an **optimistic estimate** of in-distribution performance, not an independent test score.
- **In-Distribution Validation Accuracy**: **87.39%** (97/111) [Wilson 95% CI: 79.9% – 92.3%].
- **Validation Human Recall**: **91.9%** (34/37 correct; 3 false positives flipped to AI by the PPL rule).
- **Validation AI Recall**: **94.6%** (35/37 correct).
- **Validation Paraphrased Recall**: **75.7%** (28/37 correct).

### Out-of-Distribution Convenience Slice (102 samples)
- **OOD Accuracy**: **52.94% Accuracy** (54/102) [Wilson 95% CI: 43.3% – 62.3%].
- **Nature of Set**: Derived from early-stopped arrow streaming (`batch_idx < 30`) and represents an opportunistic convenience slice rather than a balanced multi-domain benchmark. It is heavily skewed: **82.4% reviews** (84/102).
- Only the `reviews` domain contains all three classes (47/84 = 56.0% accuracy). `poetry` (10 samples) and `recipes` (8 samples) contain exclusively paraphrased-class samples.

### PPL < 28 Rule Ablation
- **In-Distribution Validation Set**: No-rule ensemble achieved **90.09%** (100/111) vs. with-rule **87.39%** (97/111). The rule wrongly flipped 3 low-perplexity human texts into AI, degrading accuracy by 2.7%.
- **Out-of-Distribution**: No-rule ensemble achieved **49.0%** (50/102) vs. with-rule **52.9%** (54/102), a 4-sample shift that is within statistical noise. In OOD, the rule flipped 5 paraphrased samples and 2 human samples (0 true AI samples).

### Hand-Written 11-Case Sanity Check (`test_hallucination.py`)
- **Nature of Script**: A hand-written smoke test / sanity check comprising 11 short text snippets (7 Human, 3 AI, 1 Paraphrased).
- **Not Accuracy Evidence**: Evaluated against a divergent, non-production pipeline (uses a 60/40 weighting rather than 65/35, and includes hardcoded heuristic regex overrides that force `HUMAN-WRITTEN` if PPL > 500 or if informal words/lowercase `i` are present). It serves purely as an engineering smoke test, not as statistical evidence of generalization.

### ⚠️ Project Limitations & Academic Failure Modes
1. **Validation-Tuned Estimates**: The 111-sample set is a validation split (37/class) used for model checkpointing and tuning. Approximately 35% derives from `data_corpus.py` (39/111) and 65% from the curated RAID slice (72/111). True generalization on completely untouched data is lower.
2. **OOD Set Composition**: The 102-sample OOD set is an early-stopped convenience slice (82% reviews). It is not domain-balanced.
3. **OOD Generator Limitation**: The FULLY AI-GENERATED class in the OOD benchmark consists exclusively of Mistral-family outputs (24 Mistral base, 10 Mistral-Chat) from the reviews domain.
4. **Class 2 / Paraphrase Label Ambiguities**: 16 samples labeled PARAPHRASED in the OOD benchmark have `generator="human"`. These represent human source texts subjected to automated paraphrase or synonym substitution attacks in the RAID benchmark.
5. **Heuristic PPL Rule Failure**: The PPL < 28 rule is a heuristic that failed on human text in-distribution, incorrectly flagging 3 out of 37 human texts as AI.
6. **128-Token Truncation**: Sequence representations are truncated to 128 tokens for CPU efficiency, meaning text beyond ~100 words is not modeled.
7. **Leakage & Overlap**: Confirmed no exact-string overlap between the 102 OOD texts and 444 training texts. TF-IDF n-gram cosine similarity check showed 0 pairs with similarity $\ge 0.60$ (maximum observed similarity was 0.366).


### `image_model.pt` — ImageDeepfakeDetector
- **Backbone**: EfficientNet-B0 (frozen, ImageNet pretrained)
- **Status**: Run `python train_image.py` with real data (data/real/ and data/fake/)
- **Recommended datasets**: FaceForensics++ or Celeb-DF

---

## 📦 All Dependencies (requirements.txt)

```
fastapi, uvicorn            -> Web framework + ASGI server
torch, torchvision          -> PyTorch + EfficientNet CNN
transformers                -> DeBERTa v3, DistilGPT-2
nltk                        -> NER, POS, sentiment, tokenization
scikit-learn                -> Random Forest, F1/confusion matrix
mlflow                      -> Experiment tracking + model registry
celery, redis               -> Async task queue + message broker
prometheus-fastapi-instrumentator -> /metrics endpoint
slowapi                     -> Rate limiting middleware
opencv-python-headless      -> Face detection (Haar Cascade)
Pillow                      -> Image loading/processing
SQLAlchemy, psycopg2-binary -> PostgreSQL / SQLite ORM
PyPDF2, python-docx         -> Document parsing (PDF, DOCX)
websockets, httpx           -> WebSocket + async HTTP client
evidently                   -> Data drift visualization
pandas, numpy               -> Data processing
sentencepiece               -> DeBERTa tokenizer dependency
python-multipart            -> FastAPI file upload support
```

---

## 🗓️ Development History

| Date | What was done |
|---|---|
| Initial | Text deepfake detector — DeBERTa + RF ensemble, Celery, MLflow, Prometheus, Docker, Chrome Extension |
| Oct 31, Session 2 | Added NER, N-gram entropy, sentiment, POS tagging to `nlp_engine.py` |
| Oct 31, Session 2 | Added `/nlp/analyze` API endpoint exposing all NLP features |
| Oct 31, Session 2 | Added `BatchNorm1d`, `GELU`, deeper fusion head to `model.py` |
| Oct 31, Session 2 | Enhanced `train.py`: `CosineAnnealingLR`, gradient clipping, F1/confusion matrix, MLflow logging |
| Oct 31, Session 2 | Added `WebSocket /ws/predict`, rate limiting (SlowAPI), `/admin/retrain` to `main.py` |
| Oct 31, Session 2 | Created `nginx.conf` (reverse proxy + load balancer + WebSocket upgrade) |
| Oct 31, Session 2 | Added Nginx service to `docker-compose.yml` |
| Oct 31, Session 2 | Created `.github/workflows/ci_cd.yml` — GitHub Actions CI/CD pipeline |
| Oct 31, Session 2 | Created `network_demo.py` — TCP socket, DNS, HTTP protocol, OSI model demo |
| Oct 31, Session 2 | Created `image_engine.py` — EfficientNet-B0 CNN + 2D FFT + Grad-CAM |
| Oct 31, Session 2 | Created `train_image.py` — image training with data augmentation + MLflow Registry |
| Oct 31, Session 2 | Added `POST /predict/image` endpoint to `main.py` |
| Oct 31, Session 2 | Updated `requirements.txt`: added slowapi, websockets, torchvision, Pillow, opencv |
| Oct 31, Session 3 | Added graceful SQLite fallback (`feedback.db`) in `main.py` when PostgreSQL is unreachable |
| Oct 31, Session 3 | Installed `torchvision` (0.29.2) & `opencv-python-headless` (5.0.0) for image engine |
| Oct 31, Session 3 | Added self-downloading NLTK chunker/corpus resources in `nlp_engine.py` for NER & POS tagging |
| Oct 31, Session 3 | Verified end-to-end local FastAPI server startup and HTTP 200 response on `/docs` and `/` |
| Oct 31, Session 4 | Synchronized File Map (`test_inference.py`, `feedback.db`, `static/index.html`) & added Windows Celery `-P solo` notes |
| Oct 31, Session 5 | Cleaned `.kilo` & `__pycache__`, decoupled image module into `image_module/`, strictly focused root on primary text detection |
| Oct 31, Session 6 | Deep multi-domain training on 90 samples (`data_corpus.py`), Focal Loss + CosineAnnealingLR (70 epochs), 94.44% Val F1, 100% on hallucination stress test |
| Nov 01, Session 7 | Fixed "Mixed/Unknown" label bug in tasks.py and index.html with explicit per-model labels; added startup model warm-up & non-blocking Celery probe (<50ms fallback); added 5 Demo Preset Scenarios, Actionable Risk Recommendation Banner, Linguistic Diagnostics (Sentiment, Entropy, POS), and Export Audit Certificate (.txt) to web UI; verified 7/7 end-to-end tests (Single Text, TXT Batch, DOCX Batch, /nlp/analyze, /metrics, Chrome Extension V3) |
| Nov 01, Session 8 | Fixed critical false-positive bug where English prose with semicolons and 'let' was detected as code (ppl=600); replaced naive keyword checks with strict code regex in nlp_engine.py; added low-PPL (<30) LLM calibration in tasks.py; verified ChatGPT self-help prompt classifies as HUMAN-POLISHED / AI-PARAPHRASED (82.0% conf, PPL=21.03) |
| Nov 01, Session 9 | (SUPERSEDED earlier prototype; not comparable to final results) Expanded dataset to 195 multi-domain samples; retrained DeBERTa dual-branch + Random Forest with Focal Loss + CosineAnnealingLR (89.7% on a very small validation split, so the 100% recall figures were not statistically meaningful); calibrated tasks.py (65% DeBERTa + 35% RF + low-PPL rule); fixed Chrome extension box coloring bug; smoke-tested the live API on 3 hand-picked texts (ChatGPT, Human, Paraphrased), which is a demo and not an accuracy measure. |

