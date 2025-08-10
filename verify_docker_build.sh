#!/bin/bash

echo "🔍 Verifying Docker build setup..."

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
    echo "📋 Contents:"
    head -5 requirements.txt
else
    echo "❌ requirements.txt not found!"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✅ Docker is available"
else
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is available"
else
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo ""
echo "🚀 Starting Docker build test..."

# Test Docker build
echo "Building Docker image..."
docker build -t whisper-diarization-test .

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful!"
    echo ""
    echo "🧪 Testing Docker Compose..."
    
    # Test docker-compose
    docker-compose config
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker Compose configuration is valid!"
        echo ""
        echo "🎉 All checks passed! You can now run:"
        echo "   docker-compose up -d"
    else
        echo "❌ Docker Compose configuration has issues"
    fi
else
    echo "❌ Docker build failed!"
    echo "Check the error messages above for details."
    exit 1
fi
