import os
import uuid
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from celery import Celery

from diarize import process_audio_file
from helpers import cleanup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Whisper Diarization API",
    description="API for audio transcription and speaker diarization using Whisper and NeMo",
    version="1.0.0"
)

# Celery configuration
celery_app = Celery(
    "whisper_diarization",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    result_expires=3600,  # Results expire in 1 hour
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_backend_transport_options={
        'retry_policy': {
            'timeout': 5.0
        }
    },
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True
)

# Pydantic models
class DiarizationRequest(BaseModel):
    audio_file: str
    language: Optional[str] = None
    whisper_model: str = "large-v3"     
    batch_size: int = 8
    device: str = "cuda"
    stemming: bool = True
    suppress_numerals: bool = False

class DiarizationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

# Storage for task results (in production, use Redis or database)
task_results = {}

@celery_app.task(bind=True)
def process_audio_task(self, audio_file_path: str, output_dir: str, **kwargs):
    """Celery task for processing audio file"""
    try:
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Starting processing...'})
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the audio file
        self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100, 'status': 'Transcribing audio...'})
        
        # Call the diarization function
        process_audio_file(audio_file_path, output_dir, **kwargs)
        
        self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100, 'status': 'Generating outputs...'})
        
        # Generate output files
        base_name = Path(audio_file_path).stem
        txt_file = os.path.join(output_dir, f"{base_name}.txt")
        srt_file = os.path.join(output_dir, f"{base_name}.srt")
        
        # Read results
        transcript = ""
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
        
        self.update_state(state='SUCCESS', meta={
            'current': 100,
            'total': 100,
            'status': 'Processing completed',
            'result': {
                'transcript': transcript,
                'txt_file': txt_file,
                'srt_file': srt_file,
                'output_dir': output_dir
            }
        })
        
        return {
            'transcript': transcript,
            'txt_file': txt_file,
            'srt_file': srt_file,
            'output_dir': output_dir
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing audio: {error_msg}")
        # Let Celery handle the exception properly
        raise

@app.post("/upload", response_model=DiarizationResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    whisper_model: str = "large-v3",
    batch_size: int = 8,
    device: str = "cuda",
    stemming: bool = True,
    suppress_numerals: bool = False
):
    """Upload audio file for diarization"""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Create unique task ID
    task_id = str(uuid.uuid4())
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = uploads_dir / f"{task_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Create output directory
    output_dir = Path("outputs") / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start background task
    task = process_audio_task.delay(
        str(file_path),
        str(output_dir),
        language=language,
        whisper_model=whisper_model,
        batch_size=batch_size,
        device=device,
        stemming=stemming,
        suppress_numerals=suppress_numerals
    )
    
    # Store task info
    task_results[task_id] = {
        'celery_task_id': task.id,
        'status': 'PENDING',
        'file_path': str(file_path),
        'output_dir': str(output_dir)
    }
    
    return DiarizationResponse(
        task_id=task_id,
        status="PENDING",
        message="Audio file uploaded and processing started"
    )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a diarization task"""
    
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_results[task_id]
    celery_task_id = task_info['celery_task_id']
    
    try:
        # Get Celery task status
        task = celery_app.AsyncResult(celery_task_id)
        
        if task.state == 'PENDING':
            status = 'PENDING'
            result = None
            error = None
        elif task.state == 'PROGRESS':
            status = 'PROCESSING'
            result = task.info
            error = None
        elif task.state == 'SUCCESS':
            status = 'COMPLETED'
            result = task.result
            error = None
            # Update local status
            task_results[task_id]['status'] = 'COMPLETED'
        elif task.state == 'FAILURE':
            status = 'FAILED'
            result = None
            error = str(task.info) if task.info else "Task failed"
            # Update local status
            task_results[task_id]['status'] = 'FAILED'
        else:
            status = 'UNKNOWN'
            result = None
            error = None
        
        return TaskStatus(
            task_id=task_id,
            status=status,
            result=result,
            error=error
        )
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {str(e)}")
        return TaskStatus(
            task_id=task_id,
            status='ERROR',
            result=None,
            error=f"Failed to get task status: {str(e)}"
        )

@app.get("/download/{task_id}")
async def download_results(task_id: str):
    """Download the results of a completed task"""
    
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_results[task_id]
    
    if task_info['status'] != 'COMPLETED':
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    output_dir = Path(task_info['output_dir'])
    
    # Check if output files exist
    txt_files = list(output_dir.glob("*.txt"))
    srt_files = list(output_dir.glob("*.srt"))
    
    if not txt_files and not srt_files:
        raise HTTPException(status_code=404, detail="No output files found")
    
    # Return file paths for download
    return {
        "task_id": task_id,
        "output_directory": str(output_dir),
        "available_files": {
            "txt_files": [str(f) for f in txt_files],
            "srt_files": [str(f) for f in srt_files]
        }
    }

@app.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """Clean up task files and results"""
    
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_results[task_id]
    
    try:
        # Clean up files
        if os.path.exists(task_info['file_path']):
            os.remove(task_info['file_path'])
        
        if os.path.exists(task_info['output_dir']):
            cleanup(task_info['output_dir'])
        
        # Remove task from memory
        del task_results[task_id]
        
        return {"message": f"Task {task_id} cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Redis is accessible
        redis_healthy = False
        try:
            celery_app.control.inspect().active()
            redis_healthy = True
        except Exception:
            redis_healthy = False
        
        return {
            "status": "healthy" if redis_healthy else "degraded",
            "service": "whisper-diarization-api",
            "redis": "connected" if redis_healthy else "disconnected",
            "celery": "available" if redis_healthy else "unavailable"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "whisper-diarization-api",
            "error": str(e)
        }

@app.get("/models")
async def get_models():
    """Get information about available Whisper models"""
    return {
        "available_models": settings.get_available_models(),
        "multilingual_models": settings.get_multilingual_models(),
        "english_models": settings.get_english_models(),
        "model_details": settings.WHISPER_MODELS,
        "default_model": settings.DEFAULT_WHISPER_MODEL,
        "recommendations": {
            "fast_processing": ["tiny", "tiny.en", "base", "base.en"],
            "balanced": ["small", "small.en"],
            "medium_accuracy": ["medium", "medium.en"],
            "high_accuracy": ["large", "large-v1", "large-v2", "large-v3"]
        }
    }

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific Whisper model"""
    model_info = settings.get_model_info(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return {
        "model_name": model_name,
        "info": model_info,
        "comparison": {
            "size_mb": float(model_info.get("size", "0 MB").replace(" MB", "")),
            "parameters": model_info.get("parameters", "0M"),
            "multilingual": model_info.get("multilingual", False),
            "accuracy_level": model_info.get("accuracy", "Unknown"),
            "speed_level": model_info.get("speed", "Unknown")
        }
    }

@app.get("/models/recommend")
async def get_model_recommendations(
    accuracy: Optional[str] = None,
    speed: Optional[str] = None,
    multilingual: Optional[bool] = None,
    max_size_mb: Optional[float] = None
):
    """Get model recommendations based on criteria"""
    recommendations = []
    
    for model_name, model_info in settings.WHISPER_MODELS.items():
        include = True
        
        # Filter by accuracy
        if accuracy:
            accuracy_levels = {"Low": 1, "Moderate": 2, "Good": 3, "High": 4, "Very High": 5}
            min_level = accuracy_levels.get(accuracy, 1)
            model_level = accuracy_levels.get(model_info.get("accuracy", "Low"), 1)
            if model_level < min_level:
                include = False
        
        # Filter by speed
        if speed:
            speed_levels = {"Very Fast": 1, "Fast": 2, "Medium": 3, "Slow": 4, "Very Slow": 5}
            max_level = speed_levels.get(speed, 5)
            model_level = speed_levels.get(model_info.get("speed", "Very Slow"), 5)
            if model_level > max_level:
                include = False
        
        # Filter by multilingual
        if multilingual is not None:
            if model_info.get("multilingual", False) != multilingual:
                include = False
        
        # Filter by size
        if max_size_mb:
            model_size = float(model_info.get("size", "0 MB").replace(" MB", ""))
            if model_size > max_size_mb:
                include = False
        
        if include:
            recommendations.append({
                "model_name": model_name,
                "info": model_info
            })
    
    return {
        "criteria": {
            "accuracy": accuracy,
            "speed": speed,
            "multilingual": multilingual,
            "max_size_mb": max_size_mb
        },
        "recommendations": recommendations,
        "total_found": len(recommendations)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
