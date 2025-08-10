import os
import uuid
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
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
    description="API for audio transcription and speaker diarization using Whisper and NeMo. Advanced parameters are automatically configured for optimal performance.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Audio Processing",
            "description": "Upload and process audio files for transcription and diarization"
        }
    ]
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

@app.post("/upload", response_model=DiarizationResponse, 
          summary="Upload Audio for Diarization",
          description="Upload an audio file to start the transcription and speaker diarization process. Advanced settings (model, batch size, device) are automatically configured for optimal performance and are not exposed in the API.")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to process (MP3, WAV, etc.)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es', 'fr'). If not specified, Whisper will auto-detect."),
    stemming: bool = Form(True, description="Enable source separation to isolate different audio sources (speakers, music, etc.)"),
    suppress_numerals: bool = Form(False, description="Convert numbers to written form (e.g., '123' becomes 'one hundred twenty three')")
):
    
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
    
    # Start background task with default values for hidden parameters
    task = process_audio_task.delay(
        str(file_path),
        str(output_dir),
        language=language,
        whisper_model="large-v3",  # Default value
        batch_size=8,               # Default value
        device="cuda",              # Default value
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

@app.get("/status/{task_id}", response_model=TaskStatus,
         summary="Get Task Status",
         description="Check the current status and progress of a diarization task")
async def get_task_status(task_id: str):
    
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

@app.get("/download/{task_id}",
         summary="Download Results",
         description="Download the transcription and diarization results for a completed task")
async def download_results(task_id: str):
    
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
