# Whisper Diarization API

A FastAPI-based REST API for audio transcription and speaker diarization using OpenAI's Whisper and NVIDIA's NeMo models.

## Features

- **Audio Upload**: Upload audio files for processing
- **Background Processing**: Asynchronous processing using Celery
- **Multiple Output Formats**: Generate both text transcripts and SRT subtitle files
- **Speaker Diarization**: Identify and separate different speakers in audio
- **Language Detection**: Automatic language detection or manual specification
- **GPU Support**: CUDA support for faster processing
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Architecture

- **FastAPI**: Modern, fast web framework for building APIs
- **Celery**: Distributed task queue for background processing
- **Redis**: Message broker and result backend for Celery
- **Whisper**: OpenAI's speech recognition model
- **NeMo**: NVIDIA's speaker diarization model

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd whisper-diarization
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

3. **Access the API**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Redis**
   ```bash
   # macOS
   brew install redis
   
   # Ubuntu/Debian
   sudo apt-get install redis-server
   ```

3. **Start the services**
   ```bash
   ./start_dev.sh
   ```

## API Endpoints

### 1. Upload Audio File
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Audio file (required)
- language: Language code (optional)
- whisper_model: Whisper model name (default: "medium.en")
- batch_size: Batch size for processing (default: 8)
- device: Processing device (default: "cpu")
- stemming: Enable source separation (default: true)
- suppress_numerals: Suppress numerical digits (default: false)
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "PENDING",
  "message": "Audio file uploaded and processing started"
}
```

### 2. Check Task Status
```http
GET /status/{task_id}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "PROCESSING",
  "result": {
    "current": 50,
    "total": 100,
    "status": "Transcribing audio..."
  },
  "error": null
}
```

### 3. Download Results
```http
GET /download/{task_id}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "output_directory": "/app/outputs/uuid-string",
  "available_files": {
    "txt_files": ["/app/outputs/uuid-string/audio.txt"],
    "srt_files": ["/app/outputs/uuid-string/audio.srt"]
  }
}
```

### 4. Cleanup Task
```http
DELETE /cleanup/{task_id}
```

**Response:**
```json
{
  "message": "Task uuid-string cleaned up successfully"
}
```

### 5. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "whisper-diarization-api"
}
```

## Usage Examples

### Python Client

```python
import requests
import time

# Upload audio file
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    data = {
        'language': 'en',
        'whisper_model': 'medium.en',
        'device': 'cpu'
    }
    response = requests.post('http://localhost:8000/upload', files=files, data=data)
    task_id = response.json()['task_id']

# Check status
while True:
    status_response = requests.get(f'http://localhost:8000/status/{task_id}')
    status_data = status_response.json()
    
    if status_data['status'] == 'COMPLETED':
        print("Processing completed!")
        break
    elif status_data['status'] == 'FAILED':
        print(f"Processing failed: {status_data['error']}")
        break
    
    print(f"Status: {status_data['status']}")
    time.sleep(5)

# Download results
download_response = requests.get(f'http://localhost:8000/download/{task_id}')
print(download_response.json())
```

### cURL Examples

**Upload audio:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "whisper_model=medium.en"
```

**Check status:**
```bash
curl -X GET "http://localhost:8000/status/{task_id}" \
  -H "accept: application/json"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_WORKER_CONCURRENCY=1

# Model Configuration
DEFAULT_WHISPER_MODEL=medium.en
DEFAULT_BATCH_SIZE=8
DEFAULT_DEVICE=cpu
```

### Whisper Models

Available Whisper models:
- `tiny.en` / `tiny` - Fastest, least accurate
- `base.en` / `base` - Fast, good accuracy
- `small.en` / `small` - Balanced speed/accuracy
- `medium.en` / `medium` - Good accuracy, moderate speed
- `large` - Best accuracy, slowest

## Docker Commands

### Build and Run
```bash
# Build the image
docker build -t whisper-diarization .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Services
```bash
# Start only Redis
docker-compose up redis -d

# Start Celery worker
docker-compose up celery-worker -d

# Start API
docker-compose up api -d
```

## Development

### Project Structure
```
whisper-diarization/
├── app.py                 # FastAPI application
├── diarize.py            # Core diarization logic
├── helpers.py            # Helper functions
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker services orchestration
├── start_dev.sh          # Development startup script
└── README_API.md         # This file
```

### Adding New Features

1. **New API Endpoints**: Add to `app.py`
2. **Background Tasks**: Create new Celery tasks in `app.py`
3. **Configuration**: Update `config.py`
4. **Dependencies**: Add to `requirements.txt`

### Testing

```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=app
```

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running: `redis-cli ping`
   - Check Redis configuration in `config.py`

2. **CUDA/GPU Issues**
   - Verify CUDA installation: `nvidia-smi`
   - Set `device=cpu` for CPU-only processing

3. **Memory Issues**
   - Reduce `batch_size` parameter
   - Use smaller Whisper model
   - Process shorter audio files

4. **File Upload Issues**
   - Check file size limits
   - Verify audio file format
   - Ensure uploads directory exists

### Logs

```bash
# View API logs
docker-compose logs api

# View Celery worker logs
docker-compose logs celery-worker

# View Redis logs
docker-compose logs redis
```

## Performance Tuning

### GPU Optimization
- Use CUDA device when available
- Adjust batch size based on GPU memory
- Use appropriate Whisper model size

### Memory Management
- Process files in chunks for large audio files
- Clean up temporary files regularly
- Monitor memory usage during processing

### Scaling
- Increase Celery worker concurrency
- Use multiple Celery workers
- Implement load balancing for multiple API instances

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation at `/docs`
