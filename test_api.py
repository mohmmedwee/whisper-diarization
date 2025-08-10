#!/usr/bin/env python3
"""
Test script for the Whisper Diarization API
"""

import requests
import time
import json
from pathlib import Path

# API base URL
API_BASE = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is it running?")
        return False

def test_upload(audio_file_path):
    """Test file upload endpoint"""
    print(f"Testing file upload with {audio_file_path}...")
    
    if not Path(audio_file_path).exists():
        print(f"✗ Audio file not found: {audio_file_path}")
        return None
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'language': 'en',
                'whisper_model': 'large-v3',  # Use latest and best model
                'device': 'cpu'
            }
            
            response = requests.post(f"{API_BASE}/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                task_id = result['task_id']
                print(f"✓ Upload successful. Task ID: {task_id}")
                return task_id
            else:
                print(f"✗ Upload failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"✗ Upload error: {str(e)}")
        return None

def test_status(task_id):
    """Test status checking endpoint"""
    print(f"Testing status check for task {task_id}...")
    
    try:
        response = requests.get(f"{API_BASE}/status/{task_id}")
        
        if response.status_code == 200:
            status_data = response.json()
            print(f"✓ Status check successful: {status_data['status']}")
            return status_data
        else:
            print(f"✗ Status check failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Status check error: {str(e)}")
        return None

def test_download(task_id):
    """Test download endpoint"""
    print(f"Testing download for task {task_id}...")
    
    try:
        response = requests.get(f"{API_BASE}/download/{task_id}")
        
        if response.status_code == 200:
            download_data = response.json()
            print("✓ Download check successful")
            print(f"  Output directory: {download_data['output_directory']}")
            print(f"  Available files: {download_data['available_files']}")
            return download_data
        else:
            print(f"✗ Download check failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Download check error: {str(e)}")
        return None

def test_cleanup(task_id):
    """Test cleanup endpoint"""
    print(f"Testing cleanup for task {task_id}...")
    
    try:
        response = requests.delete(f"{API_BASE}/cleanup/{task_id}")
        
        if response.status_code == 200:
            cleanup_data = response.json()
            print("✓ Cleanup successful")
            return True
        else:
            print(f"✗ Cleanup failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Cleanup error: {str(e)}")
        return False

def monitor_task(task_id, max_wait_time=300):
    """Monitor a task until completion or timeout"""
    print(f"Monitoring task {task_id}...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status_data = test_status(task_id)
        
        if not status_data:
            return False
        
        if status_data['status'] == 'COMPLETED':
            print("✓ Task completed successfully!")
            return True
        elif status_data['status'] == 'FAILED':
            print(f"✗ Task failed: {status_data.get('error', 'Unknown error')}")
            return False
        elif status_data['status'] == 'PROCESSING':
            progress = status_data.get('result', {})
            current = progress.get('current', 0)
            total = progress.get('total', 100)
            status_msg = progress.get('status', 'Processing...')
            print(f"  Progress: {current}/{total} - {status_msg}")
        
        time.sleep(5)
    
    print("✗ Task monitoring timed out")
    return False

def main():
    """Main test function"""
    print("Whisper Diarization API Test")
    print("=" * 40)
    
    # Test health check
    if not test_health():
        print("\nCannot proceed without a healthy API. Please start the services.")
        return
    
    print("\nAPI is healthy. Starting tests...")
    
    # Test with a sample audio file
    audio_file = "tests/assets/test.opus"  # Use the existing test file
    
    if not Path(audio_file).exists():
        print(f"\nNo test audio file found at {audio_file}")
        print("Please provide a valid audio file path as an argument or place one in tests/assets/")
        return
    
    # Test upload
    task_id = test_upload(audio_file)
    if not task_id:
        print("\nUpload test failed. Cannot continue.")
        return
    
    print(f"\nTask {task_id} created successfully.")
    
    # Monitor the task
    if monitor_task(task_id, max_wait_time=600):  # 10 minutes timeout
        # Test download
        test_download(task_id)
        
        # Test cleanup
        test_cleanup(task_id)
    else:
        print("Task did not complete successfully.")
    
    print("\nTest completed!")

def test_models():
    """Test the models endpoint"""
    print("Testing models endpoint...")
    
    try:
        response = requests.get(f"{API_BASE}/models")
        
        if response.status_code == 200:
            models_data = response.json()
            print("✓ Models endpoint successful")
            print(f"  Available models: {len(models_data['available_models'])}")
            print(f"  Multilingual models: {len(models_data['multilingual_models'])}")
            print(f"  English models: {len(models_data['english_models'])}")
            print(f"  Default model: {models_data['default_model']}")
            return models_data
        else:
            print(f"✗ Models endpoint failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Models endpoint error: {str(e)}")
        return None

def test_model_info(model_name):
    """Test the specific model info endpoint"""
    print(f"Testing model info for {model_name}...")
    
    try:
        response = requests.get(f"{API_BASE}/models/{model_name}")
        
        if response.status_code == 200:
            model_data = response.json()
            print(f"✓ Model info successful for {model_name}")
            print(f"  Size: {model_data['info']['size']}")
            print(f"  Parameters: {model_data['info']['parameters']}")
            print(f"  Accuracy: {model_data['info']['accuracy']}")
            print(f"  Speed: {model_data['info']['speed']}")
            return model_data
        else:
            print(f"✗ Model info failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Model info error: {str(e)}")
        return None

def test_model_recommendations():
    """Test the model recommendations endpoint"""
    print("Testing model recommendations...")
    
    try:
        # Test different recommendation scenarios
        test_cases = [
            {"accuracy": "Good", "speed": "Medium"},
            {"multilingual": True, "max_size_mb": 500},
            {"speed": "Fast", "max_size_mb": 100}
        ]
        
        for i, criteria in enumerate(test_cases, 1):
            print(f"  Test case {i}: {criteria}")
            response = requests.get(f"{API_BASE}/models/recommend", params=criteria)
            
            if response.status_code == 200:
                rec_data = response.json()
                print(f"    ✓ Found {rec_data['total_found']} recommendations")
                for rec in rec_data['recommendations'][:3]:  # Show first 3
                    print(f"      - {rec['model_name']}: {rec['info']['size']}, {rec['info']['accuracy']}")
            else:
                print(f"    ✗ Failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model recommendations error: {str(e)}")
        return False

def test_all_model_endpoints():
    """Test all model-related endpoints"""
    print("\n" + "=" * 50)
    print("Testing Model Endpoints")
    print("=" * 50)
    
    # Test models list
    models_data = test_models()
    if not models_data:
        print("Models endpoint test failed. Skipping other model tests.")
        return
    
    # Test specific model info
    if models_data['available_models']:
        test_model = models_data['available_models'][0]
        test_model_info(test_model)
    
    # Test recommendations
    test_model_recommendations()
    
    print("Model endpoints testing completed!")

if __name__ == "__main__":
    main()
    
    # Uncomment the line below to test model endpoints separately
    # test_all_model_endpoints()
