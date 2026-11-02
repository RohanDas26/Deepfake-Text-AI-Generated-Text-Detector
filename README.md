# Multi-Modal AI Text Deepfake Detection & Attribution System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Celery-5.3%2B-37814A?logo=celery&logoColor=white" alt="Celery" />
  <img src="https://img.shields.io/badge/Redis-In--Memory-DC382D?logo=redis&logoColor=white" alt="Redis" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

---

## Executive Summary

The proliferation of Large Language Models (LLMs) like GPT-4, Claude, and Llama has made synthetic text virtually indistinguishable from human writing. Furthermore, simple adversarial rewriters (e.g., QuillBot) readily bypass conventional single-metric detectors.

This project delivers a **production-grade, 3-class AI deepfake text detection system** that classifies content into:
1. **Human-Written:** Naturally authored human text.
2. **Fully AI-Generated:** Unmodified machine generations.
3. **Human-Polished / AI-Paraphrased:** Text altered by adversarial paraphrase or synonym perturbation attacks.

Rather than relying on an opaque transformer or fragile heuristic rules alone, our system couples a **frozen DeBERTa-v3 contextual backbone** with an **8-dimensional statistical NLP feature extraction engine** through a **dual-branch neural fusion network**, ensembled with a **Random Forest classifier (65/35 weighted blend)** and served via an **asynchronous MLOps pipeline**.

---

## Academic Subject Mapping

This project represents the joint capstone implementation for three core computer science disciplines:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-MODAL DETECTION SYSTEM                          │
├─────────────────────┬───────────────────────────┬───────────────────────────┤
│    NLP DISCIPLINE   │      PDNN DISCIPLINE      │      MLOPS DISCIPLINE     │
├─────────────────────┼───────────────────────────┼───────────────────────────┤
│ • Perplexity Loss   │ • DeBERTa-v3 Transformer  │ • Celery Distributed Tasks│
│ • Sentence Burst    │ • Dual-Branch Fusion MLP  │ • Redis Message Broker    │
│ • Type-Token Ratio  │ • Focal Loss Optimization │ • MLflow Experiment Store │
│ • N-Gram Entropy    │ • BatchNorm1d Regularize  │ • Prometheus Live Metrics │
│ • VADER Sentiment   │ • CosineAnnealingLR       │ • Docker Containerization │
│ • POS Noun/Verb Tag │ • Weighted Soft Ensemble  │ • Chrome Extension Client │
└─────────────────────┴───────────────────────────┴───────────────────────────┘
```

---

## System Architecture

```
CLIENT LAYER
  [ Web Dashboard ]        [ Chrome Extension ]        [ REST / WebSocket Clients ]
         │                          │                               │
         └──────────────────────────┴───────────────────────────────┘
                                    │ HTTP / WebSocket (Port 8000)
                                    ▼
APPLICATION LAYER
  FastAPI Server (Rate Limiting, Token Bucket, Request Routing, CORS)
         │
         ├─── Dispatches Async Task ────► Redis Broker (Port 6379)
         └─── Live Metrics Scrape   ────► Prometheus (Port 9090)
                                                 │
WORKER & INFERENCE LAYER                         ▼
  Celery Worker Node (tasks.py) ◄────────────────┘
         │
         ├── Branch 1: DeBERTa-v3 Small Contextual Embeddings (768-dim)
         ├── Branch 2: NLPEngine Statistical Feature Vector (8-dim)
         │       └─ Projected to 32-dim (Linear -> BatchNorm1d -> GELU)
         │
         ├── Neural Fusion Head: Concat(768 + 32 = 800-dim) -> 3 Logits
         ├── Classical ML Head: Random Forest Classifier (8-dim NLP Input)
         │
         └── Soft-Voting Ensemble: (0.65 * DeBERTa) + (0.35 * Random Forest)
                                    │
                                    ▼
OBSERVABILITY & PERSISTENCE
  ├── MLflow Registry (Port 5000) ── Experiment tracking & parameter versioning
  ├── Prometheus & Grafana        ── Latency, throughput, and error rates
  └── Inference Drift Log         ── JSONL streaming log for concept drift detection
```

---

## Interactive Feature Breakdown

<details>
<summary><b>1. Natural Language Processing (NLP Engine)</b></summary>
<br>

The `NLPEngine` (`nlp_engine.py`) extracts an 8-dimensional mathematical profile of text style independent of token identity:

1. **Log Perplexity ($\log(\text{PPL})$):** Computed using causal cross-entropy loss from `distilgpt2`:
   $$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_{<i}) \right)$$
   AI text exhibits low, predictable perplexity; human writing produces elevated variance.
2. **Burstiness:** The coefficient of variation of sentence lengths ($\sigma / \mu$):
   $$\text{Burstiness} = \frac{\sigma_{\text{sentence\_len}}}{\mu_{\text{sentence\_len}}}$$
   Humans mix short, punchy sentences with long descriptive clauses; LLMs maintain uniform sentence length.
3. **Type-Token Ratio (TTR):** Lexical diversity metric measuring vocabulary richness:
   $$\text{TTR} = \frac{|V_{\text{unique}}|}{|V_{\text{total}}|}$$
4. **N-gram Entropy (Bigram & Trigram):** Shannon entropy of transition distributions:
   $$H(X) = -\sum_{x} P(x) \log_2 P(x)$$
   Detects repetitive, formulaic phrasing common in LLM outputs.
5. **VADER Sentiment Compound Score:** Assesses emotional tone consistency across sentences.
6. **Part-of-Speech Densities:** Noun ratio ($\text{NN}$) and verb ratio ($\text{VB}$) calculated via NLTK pos_tag.

</details>

<details>
<summary><b>2. Principles of Deep Neural Networks (PDNN Model)</b></summary>
<br>

The `PDNCDualBranchNetwork` (`model.py`) unifies deep representations with explicit linguistic rules:

- **Contextual Branch:** Frozen `microsoft/deberta-v3-small` backbone (44M parameters, `requires_grad=False`). Disentangled attention with mean pooling extracts a 768-dimensional sequence representation.
- **Statistical Projection Branch:** Projects the 8 statistical NLP features to 32 dimensions via `nn.Linear(8, 32)` followed by `nn.BatchNorm1d(32)` and `nn.GELU()`.
- **Fusion MLP Head:** Concatenates contextual and statistical features ($768 + 32 = 800$ dimensions) and routes through a dropout-regularized feedforward network:
  $$\text{Logits} = W_2 \cdot \text{GELU}(\text{Dropout}(W_1 \cdot [h_{\text{text}} \,\|\, h_{\text{stat}}] + b_1)) + b_2$$
- **Focal Loss Formulation:** Addresses class difficulty and boundary confusion:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma = 2.0)$$
- **Optimization:** AdamW optimizer ($\text{lr} = 8 \times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$) scheduled with `CosineAnnealingLR` over 70 epochs.

</details>

<details>
<summary><b>3. Classical ML Ensemble & Soft Voting</b></summary>
<br>

To guard against deep network hallucination and adversarial prompt formatting, predictions are ensembled with a tuned Random Forest classifier:
- **RF Configuration:** 100 trees, maximum depth 6, Gini impurity criterion.
- **Ensemble Blend:** Soft-voting probability combination:
  $$P_{\text{final}} = 0.65 \cdot P_{\text{DeBERTa}} + 0.35 \cdot P_{\text{RandomForest}}$$
- Combines semantic understanding with non-linear tree partitions over grammatical metrics.

</details>

<details>
<summary><b>4. Production MLOps Stack & Observability</b></summary>
<br>

- **Asynchronous Task Queue:** FastAPI pushes inference jobs to Redis; Celery worker nodes consume and execute inference without blocking the HTTP event loop.
- **Experiment Tracking:** MLflow automatically tracks hyperparameters, validation F1, and model checkpoints.
- **Real-Time Telemetry:** `prometheus-fastapi-instrumentator` exposes `/metrics` for scraping latency, throughput, and memory.
- **Data Drift Monitoring:** Inference requests are logged into `inference_logs.jsonl` to detect distribution shifts in perplexity over time.
- **Client Ecosystem:** Includes an interactive web dashboard (`static/index.html`) and an unprivileged Chrome Extension (Manifest V3) for one-click right-click browser inspection.

</details>

---

## Model Performance & Empirical Results

The system was evaluated on a **555-sample multi-domain benchmark dataset** (185 Human, 185 AI, 185 Paraphrased) combining curated samples from the RAID benchmark with a local multi-domain corpus.

### Comparative Evaluation (111-Sample Validation Holdout)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (NLP Stats Only)** | 72.07% | 0.71 | 0.72 | 0.71 | 0.84 |
| **DeBERTa-v3 Small (Contextual Only)** | 85.58% | 0.84 | 0.85 | 0.84 | 0.92 |
| **Proposed Dual-Branch Ensemble** | **87.39%** | **0.88** | **0.87** | **0.87** | **0.97** |

<details>
<summary><b>View Confusion Matrix & Per-Class Recall</b></summary>
<br>

```
                  Predicted: HUMAN     Predicted: AI     Predicted: PARAPHRASED
Actual: HUMAN            34 (91.9%)          3 (8.1%)            0 (0.0%)
Actual: AI                1 (2.7%)          35 (94.6%)           1 (2.7%)
Actual: PARAPHRASED       4 (10.8%)          5 (13.5%)          28 (75.7%)
```

- **Pure Human Recall:** 91.9% (34 of 37 correct).
- **Pure AI Recall:** 94.6% (35 of 37 correct).
- **Paraphrased Recall:** 75.7% (28 of 37 correct).
- **Academic Discussion:** Human text and AI text are cleanly distinguished. Paraphrased text remains the primary empirical challenge because adversarial synonym attacks shift statistical features toward human distributions.

</details>

---

## Team Contributions & Work Allocation

<details>
<summary><b>Click to expand 4-Member Project Division</b></summary>
<br>

| Member | Specialization | Core Modules & Deliverables |
|:---|:---|:---|
| **Rohan Das (Lead)** | NLP Feature Engineering | • Designed `NLPEngine` with 8 linguistic features.<br>• Extracted DistilGPT-2 perplexity, sentence burstiness, and TTR.<br>• Curated and balanced the 555-sample multi-domain benchmark dataset. |
| **Sai Praneetha Oruganty** | Deep Learning (PDNN) | • Implemented PyTorch `PDNCDualBranchNetwork`.<br>• Integrated frozen DeBERTa-v3 contextual backbone with mean pooling.<br>• Formulated Focal Loss ($\gamma = 2.0$) and Cosine Annealing optimization. |
| **Dhamini Nalluri** | Classical ML & Evaluation | • Trained and tuned the Random Forest baseline model (`rf_model.pkl`).<br>• Engineered the 65/35 soft-voting weighted ensemble strategy.<br>• Conducted holdout validation, confusion matrix analysis, and EDA plots. |
| **Nikhileswar Reddy** | MLOps & System Architecture | • Built FastAPI REST endpoints, rate limiting, and WebSockets.<br>• Configured Celery asynchronous task queue with Redis broker.<br>• Integrated MLflow, Prometheus metrics, and Chrome Extension client. |

</details>

---

## Quickstart Guide

### Prerequisites
- Python 3.10 or higher
- Redis (Optional for local testing; required for async queue)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/RohanDas26/Deepfake-Text-AI-Generated-Text-Detector.git
cd Deepfake-Text-AI-Generated-Text-Detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Locally (Fast Development Mode)

```bash
# Terminal 1: Start Redis (or use local Docker Redis: docker run -p 6379:6379 redis:alpine)

# Terminal 2: Start Celery worker (use -P solo on Windows)
celery -A tasks worker --loglevel=info -P solo

# Terminal 3: Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Open Web Dashboard: `http://localhost:8000`
- Open Swagger API Docs: `http://localhost:8000/docs`

### 3. Run with Docker Compose (Full MLOps Stack)

```bash
docker-compose up --build
```
- Web Application: `http://localhost:8000`
- MLflow Tracking UI: `http://localhost:5000`
- Prometheus Metrics: `http://localhost:9090`
- Grafana Dashboards: `http://localhost:3000`

---

## API Documentation

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/predict` | Submit text for asynchronous 3-class detection (returns `task_id`). |
| `GET` | `/predict/result/{task_id}` | Poll prediction results and class confidence distribution. |
| `POST` | `/predict/document` | MapReduce chunked document analysis (PDF, DOCX, TXT). |
| `GET` | `/nlp/analyze?text=...` | Full 8-dimensional linguistic feature breakdown. |
| `GET` | `/metrics` | Prometheus time-series metrics scrape endpoint. |
| `GET` | `/health` | API and worker health status check. |
| `WS` | `/ws/predict` | Real-time full-duplex WebSocket text streaming. |

<details>
<summary><b>Sample Prediction Request & Response</b></summary>
<br>

**Request (`POST /predict`):**
```json
{
  "text": "Machine learning systems rely on statistical distributions to approximate complex functions."
}
```

**Response (`GET /predict/result/{task_id}`):**
```json
{
  "task_id": "84fd8d5b-419b-466d-a7a2-f94be6bcbbd4",
  "status": "SUCCESS",
  "prediction": "HUMAN-WRITTEN",
  "confidence": 0.892,
  "probabilities": {
    "HUMAN-WRITTEN": 0.892,
    "FULLY AI-GENERATED": 0.048,
    "HUMAN-POLISHED / AI-PARAPHRASED": 0.060
  },
  "linguistic_metrics": {
    "perplexity": 42.18,
    "burstiness": 0.612,
    "ttr": 0.741,
    "sentiment": 0.12
  }
}
```

</details>

---

## Secondary Exploration Module: Image Deepfake Detection

An isolated secondary computer vision exploration module is provided in `image_module/`:
- **Architecture:** EfficientNet-B0 CNN (spatial features) + 2D Fast Fourier Transform (FFT frequency artifacts) + Grad-CAM saliency.
- **Isolation Policy:** Decoupled completely from core text dependencies to maintain clean modularity.

---

## License & Citation

This project is licensed under the MIT License.

```bibtex
@misc{das2026deepfakedetection,
  author = {Rohan Das},
  title = {Multi-Modal AI Text Deepfake Detection & Attribution System},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RohanDas26/Deepfake-Text-AI-Generated-Text-Detector}}
}
```
