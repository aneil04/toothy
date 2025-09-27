#!/usr/bin/env python3
"""
Test script to verify the YOLO model loads correctly
"""

from ultralytics import YOLO
import cv2
import numpy as np

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        print("Loading YOLO model...")
        model = YOLO("yolo_tb_finetune.pt")
        print("✅ Model loaded successfully!")
        
        # Test with a dummy image
        print("Testing inference with dummy image...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✅ Inference test passed!")
        
        # Print model info
        print(f"Model classes: {len(model.names)}")
        print("Classes:", list(model.names.values()))
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: Model file 'yolo_tb_finetune.pt' not found!")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_webcam():
    """Test if webcam is accessible"""
    try:
        print("Testing webcam access...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Webcam is accessible!")
                cap.release()
                return True
            else:
                print("❌ Could not read from webcam")
                cap.release()
                return False
        else:
            print("❌ Could not open webcam")
            return False
    except Exception as e:
        print(f"❌ Error testing webcam: {e}")
        return False

if __name__ == "__main__":
    print("=== YOLO Model Test ===")
    model_ok = test_model_loading()
    webcam_ok = test_webcam()
    
    print("\n=== Test Results ===")
    if model_ok and webcam_ok:
        print("✅ All tests passed! You can run tb_detect.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
