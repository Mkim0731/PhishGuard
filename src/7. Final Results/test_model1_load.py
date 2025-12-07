#!/usr/bin/env python3
"""Test script to check Model 1 loading"""

import sys
from pathlib import Path

# Add Model1 to path
sys.path.insert(0, str(Path(__file__).parent / 'Model1_EmailContent'))

from phishing_detection_system import PhishingDetector

# Get model path
project_root = Path(__file__).parent
model_path = project_root / 'trained_models' / 'Model1'
model_path_abs = str(model_path.resolve())

print(f"Loading Model 1 from: {model_path_abs}")
print(f"Path exists: {model_path.exists()}")

try:
    detector = PhishingDetector(require_cuda=False)
    detector.load_model(model_path_abs)
    print("✅ Model 1 loaded successfully!")
    
    # Test prediction
    test_text = "Subject: Test\nBody: This is a test email."
    result = detector.predict(test_text)
    print(f"✅ Test prediction successful: {result}")
    
except Exception as e:
    print(f"❌ Error loading Model 1: {str(e)}")
    import traceback
    traceback.print_exc()
