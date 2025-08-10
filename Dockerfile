# Use CUDA base image with CUDNN for GPU support with Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDNN_LIBRARY=/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cudnn/lib64:$LD_LIBRARY_PATH
ENV CUDNN_ROOT=/usr/local/cudnn
ENV CUDNN_INCLUDE_DIR=/usr/local/cudnn/include
ENV CUDNN_LIBRARY_DIR=/usr/local/cudnn/lib64
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    cmake \
    pkg-config \
    libfftw3-dev \
    libasound2-dev \
    libsndfile1-dev \
    libsamplerate0-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev \
    libpostproc-dev \
    libswscale-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libtheora-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDNN manually
RUN wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.9.7/local_installers/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
    apt-get update && \
    apt-get install -y libcudnn8 && \
    rm cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip install --no-cache-dir Cython numpy

# Verify CUDNN installation and create symlinks
RUN ls -la /usr/local/cudnn* || echo "CUDNN not found in expected location"
RUN find /usr/local -name "*cudnn*" -type f | head -10 || echo "No CUDNN files found"
RUN ldconfig
# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY constraints.txt .

# Install critical dependencies first
RUN pip install --no-cache-dir typing_extensions

# Install Python dependencies
RUN pip install -c constraints.txt -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs temp_outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Test CUDA/CUDNN availability
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDNN version: {torch.backends.cudnn.version()}')" || echo "CUDA test failed"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
