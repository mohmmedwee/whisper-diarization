.PHONY: help build run stop clean logs test dev docker-build docker-run docker-stop

# Default target
help:
	@echo "Whisper Diarization API - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev          - Start development environment (Redis + Celery + API)"
	@echo "  test         - Run API tests"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Start all services with Docker Compose"
	@echo "  docker-stop  - Stop all Docker services"
	@echo "  docker-logs  - View Docker logs"
	@echo ""
	@echo "Management:"
	@echo "  clean        - Clean up temporary files and outputs"
	@echo "  logs         - View application logs"

# Development commands
dev:
	@echo "Starting development environment..."
	@./start_dev.sh

test:
	@echo "Running API tests..."
	@python test_api.py

# Docker commands
docker-build:
	@echo "Building Docker image..."
	@docker build -t whisper-diarization .

docker-run:
	@echo "Starting services with Docker Compose..."
	@docker-compose up -d

docker-stop:
	@echo "Stopping Docker services..."
	@docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	@docker-compose logs -f

# Utility commands
clean:
	@echo "Cleaning up temporary files..."
	@rm -rf uploads/* outputs/* temp_outputs_*
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete

logs:
	@echo "Viewing application logs..."
	@tail -f logs/app.log 2>/dev/null || echo "No log file found"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt

# Format code
format:
	@echo "Formatting code..."
	@black *.py
	@isort *.py

# Lint code
lint:
	@echo "Linting code..."
	@flake8 *.py
	@black --check *.py
	@isort --check-only *.py
