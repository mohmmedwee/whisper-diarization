# Use CUDA base image for GPU support with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and clean up cache in a single RUN command
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    sox \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools

# Install Cython first for building dependencies
RUN pip install --no-cache-dir Cython

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with CUDA version and other dependencies
RUN pip install --no-cache-dir \
    torch==2.1.1+cu121 \
    torchaudio==2.1.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3.10 -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Create necessary directories for uploads and outputs
RUN mkdir -p uploads outputs temp_outputs


# Run the application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
