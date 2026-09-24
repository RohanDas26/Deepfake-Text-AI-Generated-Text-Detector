import logging
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from celery.result import AsyncResult
import PyPDF2
import io

from tasks import celery_app, run_inference, run_document_inference

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

@app.post("/predict/document", response_model=TaskResponse)
async def predict_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file for batch paragraph-level analysis.
    """
    if file.filename.endswith(".pdf"):
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        full_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
    elif file.filename.endswith(".txt"):
        contents = await file.read()
        full_text = contents.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document.")
        
    task = run_document_inference.delay(full_text)
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
