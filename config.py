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
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get detailed information about a specific Whisper model"""
        return cls.WHISPER_MODELS.get(model_name, {})
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of all available Whisper models"""
        return list(cls.WHISPER_MODELS.keys())
    
    @classmethod
    def get_multilingual_models(cls) -> list:
        """Get list of multilingual Whisper models"""
        return [model for model, info in cls.WHISPER_MODELS.items() 
                if info.get("multilingual", False)]
    
    @classmethod
    def get_english_models(cls) -> list:
        """Get list of English-only Whisper models"""
        return [model for model, info in cls.WHISPER_MODELS.items() 
                if not info.get("multilingual", False)]
    
    @classmethod
    def get_models_by_accuracy(cls, min_accuracy: str = "Moderate") -> list:
        """Get models with minimum accuracy level"""
        accuracy_levels = {"Low": 1, "Moderate": 2, "Good": 3, "High": 4, "Very High": 5}
        min_level = accuracy_levels.get(min_accuracy, 1)
        
        return [model for model, info in cls.WHISPER_MODELS.items() 
                if accuracy_levels.get(info.get("accuracy", "Low"), 1) >= min_level]
    
    @classmethod
    def get_models_by_speed(cls, max_speed: str = "Medium") -> list:
        """Get models with maximum speed level"""
        speed_levels = {"Very Fast": 1, "Fast": 2, "Medium": 3, "Slow": 4, "Very Slow": 5}
        max_level = speed_levels.get(max_speed, 5)
        
        return [model for model, info in cls.WHISPER_MODELS.items() 
                if speed_levels.get(info.get("speed", "Very Slow"), 5) <= max_level]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
