# Server Deployment Guide

This guide will help you deploy and test the Whisper Diarization API on your server.

## 🚀 Quick Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/mohmmedwee/whisper-diarization.git
cd whisper-diarization
```

### 2. Start with Docker Compose (Recommended)
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Verify Deployment
```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open http://your-server-ip:8000/docs in your browser
```

## 🔧 Manual Deployment

### Prerequisites
- Python 3.9+
- Redis
- CUDA (optional, for GPU support)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis
```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
# or
brew install redis                 # macOS

# Start Redis
redis-server
```

### 3. Start Celery Worker
```bash
celery -A app.celery_app worker --loglevel=info
```

### 4. Start FastAPI
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing the API

### 1. Health Check
```bash
curl http://your-server-ip:8000/health
```

### 2. Upload Audio File
```bash
curl -X POST "http://your-server-ip:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/assets/test.opus" \
  -F "language=en" \
  -F "whisper_model=tiny.en"
```

### 3. Check Task Status
```bash
# Replace {task_id} with the ID from upload response
curl -X GET "http://your-server-ip:8000/status/{task_id}"
```

### 4. Download Results
```bash
curl -X GET "http://your-server-ip:8000/download/{task_id}"
```

## 📊 Monitoring

### View Logs
```bash
# Docker logs
docker-compose logs -f api
docker-compose logs -f celery-worker
docker-compose logs -f redis

# Application logs
tail -f logs/app.log
```

### Check Service Status
```bash
# Docker services
docker-compose ps

# Redis status
redis-cli ping

# Celery status
celery -A app.celery_app inspect active
```

## 🔒 Security Considerations

### 1. Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw allow 8000  # API
sudo ufw allow 6379  # Redis (if external access needed)
```

### 2. Environment Variables
Create a `.env` file for production:
```env
# Production settings
API_HOST=0.0.0.0
API_PORT=8000
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### 3. Reverse Proxy (Optional)
```nginx
# Nginx configuration
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   sudo lsof -i :8000
   
   # Kill process
   sudo kill -9 <PID>
   ```

2. **Redis Connection Error**
   ```bash
   # Check Redis status
   sudo systemctl status redis
   
   # Restart Redis
   sudo systemctl restart redis
   ```

3. **CUDA/GPU Issues**
   ```bash
   # Check GPU status
   nvidia-smi
   
   # Use CPU mode
   export DEFAULT_DEVICE=cpu
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # Reduce batch size in config
   export DEFAULT_BATCH_SIZE=4
   ```

### Performance Tuning

1. **Increase Celery Workers**
   ```bash
   # In docker-compose.yml
   celery-worker:
     command: celery -A app.celery_app worker --loglevel=info --concurrency=4
   ```

2. **Optimize Redis**
   ```bash
   # In redis.conf
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   ```

3. **GPU Optimization**
   ```bash
   # Use larger batch size with GPU
   export DEFAULT_BATCH_SIZE=16
   export DEFAULT_DEVICE=cuda
   ```

## 📈 Scaling

### Horizontal Scaling
```bash
# Add more Celery workers
docker-compose up --scale celery-worker=3 -d

# Load balancing with multiple API instances
docker-compose up --scale api=3 -d
```

### Vertical Scaling
```bash
# Increase worker concurrency
celery -A app.celery_app worker --concurrency=8

# Use larger Whisper models
export DEFAULT_WHISPER_MODEL=large
```

## 🔄 Updates

### Pull Latest Changes
```bash
git pull origin main
docker-compose down
docker-compose up -d --build
```

### Rollback
```bash
git checkout <previous-commit>
docker-compose down
docker-compose up -d --build
```

## 📞 Support

For deployment issues:
1. Check the logs: `docker-compose logs -f`
2. Verify service status: `docker-compose ps`
3. Test individual components
4. Check the troubleshooting section above

## 🎯 Next Steps

After successful deployment:
1. Test with your audio files
2. Monitor performance and logs
3. Configure monitoring and alerting
4. Set up automated backups
5. Implement CI/CD pipeline
