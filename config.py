import os
from typing import Optional

class Settings:
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # Celery Configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    CELERY_WORKER_CONCURRENCY: int = int(os.getenv("CELERY_WORKER_CONCURRENCY", "1"))
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # File Storage
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", "uploads")
    OUTPUTS_DIR: str = os.getenv("OUTPUTS_DIR", "outputs")
    MAX_FILE_SIZE: str = os.getenv("MAX_FILE_SIZE", "100MB")
    
    # Model Configuration
    DEFAULT_WHISPER_MODEL: str = os.getenv("DEFAULT_WHISPER_MODEL", "medium.en")
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "8"))
    DEFAULT_DEVICE: str = os.getenv("DEFAULT_DEVICE", "cpu")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
