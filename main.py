import logging
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from celery import chord
from celery.result import AsyncResult
import PyPDF2
import docx
import io
import nltk

from tasks import celery_app, run_inference, analyze_chunk, aggregate_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Text Detection System - MLOps Edition")

# Initialize Prometheus Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

class PredictRequest(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

class PredictResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

@app.post("/predict", response_model=TaskResponse)
async def predict(request: PredictRequest):
    """
    Submit text for classification asynchronously via Celery worker.
    Returns a task_id to poll for results.
    """
    task = run_inference.delay(request.text)
    return {"task_id": task.id, "status": "PENDING"}

def get_sliding_chunks(text: str, window_size=3, step=2):
    """
    Intelligently chunks text into overlapping sentence windows.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    sentences = nltk.sent_tokenize(text)
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
    
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
    elif file.filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(contents))
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif file.filename.endswith(".txt"):
        full_text = contents.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
        
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document.")
        
    chunks = get_sliding_chunks(full_text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Document text too short or invalid.")
        
    # Dispatch MapReduce Celery Chord
    # Map: analyze_chunk runs in parallel across all chunks
    # Reduce: aggregate_document computes the final report
    callback = aggregate_document.s()
    header = [analyze_chunk.s(c) for c in chunks]
    
    task = chord(header)(callback)
    
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/predict/result/{task_id}", response_model=PredictResponse)
async def get_predict_result(task_id: str):
    """
    Poll this endpoint with the task_id to get the final result.
    """
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

# Serve static dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
