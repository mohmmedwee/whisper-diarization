#!/bin/bash

# Start Redis
echo "Starting Redis..."
redis-server --daemonize yes

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
sleep 3

# Start Celery worker
echo "Starting Celery worker..."
celery -A app.celery_app worker --loglevel=info --detach

# Start FastAPI application
echo "Starting FastAPI application..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
