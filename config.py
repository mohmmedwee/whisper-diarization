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
    DEFAULT_WHISPER_MODEL: str = os.getenv("DEFAULT_WHISPER_MODEL", "large-v3")
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "8"))
    DEFAULT_DEVICE: str = os.getenv("DEFAULT_DEVICE", "cuda")
    
    # Whisper Model Sizes and Configurations
    WHISPER_MODELS = {
        # Tiny models (fastest, least accurate)
        "tiny": {
            "size": "39 MB",
            "parameters": "39M",
            "multilingual": False,
            "languages": ["en"],
            "recommended_use": "Quick testing, real-time applications",
            "accuracy": "Low",
            "speed": "Very Fast"
        },
        "tiny.en": {
            "size": "39 MB",
            "parameters": "39M",
            "multilingual": False,
            "languages": ["en"],
            "recommended_use": "English-only, quick processing",
            "accuracy": "Low",
            "speed": "Very Fast"
        },
        
        # Base models (fast, moderate accuracy)
        "base": {
            "size": "74 MB",
            "parameters": "74M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "General purpose, good balance",
            "accuracy": "Moderate",
            "speed": "Fast"
        },
        "base.en": {
            "size": "74 MB",
            "parameters": "74M",
            "multilingual": False,
            "languages": ["en"],
            "recommended_use": "English-only, good balance",
            "accuracy": "Moderate",
            "speed": "Fast"
        },
        
        # Small models (balanced)
        "small": {
            "size": "244 MB",
            "parameters": "244M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "Production use, good accuracy",
            "accuracy": "Good",
            "speed": "Medium"
        },
        "small.en": {
            "size": "244 MB",
            "parameters": "244M",
            "multilingual": False,
            "languages": ["multilingual"],
            "recommended_use": "English-only production use",
            "accuracy": "Good",
            "speed": "Medium"
        },
        
        # Medium models (high accuracy)
        "medium": {
            "size": "769 MB",
            "parameters": "769M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "High accuracy requirements",
            "accuracy": "High",
            "speed": "Slow"
        },
        "medium.en": {
            "size": "769 MB",
            "parameters": "769M",
            "multilingual": False,
            "languages": ["en"],
            "recommended_use": "English-only, high accuracy",
            "accuracy": "High",
            "speed": "Slow"
        },
        
        # Large models (highest accuracy)
        "large": {
            "size": "1550 MB",
            "parameters": "1550M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "Maximum accuracy, research",
            "accuracy": "Very High",
            "speed": "Very Slow"
        },
        "large-v1": {
            "size": "1550 MB",
            "parameters": "1550M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "Maximum accuracy, research",
            "accuracy": "Very High",
            "speed": "Very Slow"
        },
        "large-v2": {
            "size": "1550 MB",
            "parameters": "1550M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "Maximum accuracy, research",
            "accuracy": "Very High",
            "speed": "Very Slow"
        },
        "large-v3": {
            "size": "1550 MB",
            "parameters": "1550M",
            "multilingual": True,
            "languages": ["en"],
            "recommended_use": "English-only, maximum accuracy",
            "accuracy": "Very High",
            "speed": "Very Slow"
        },
        
        # XLarge models (maximum accuracy)
        "xlarge": {
            "size": "3100 MB",
            "parameters": "3100M",
            "multilingual": True,
            "languages": ["multilingual"],
            "recommended_use": "Maximum accuracy, research, professional use",
            "accuracy": "Maximum",
            "speed": "Extremely Slow"
        }
    }
    
    # Model Selection Helpers
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
