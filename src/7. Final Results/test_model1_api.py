#!/usr/bin/env python3
"""
Test script to verify Model 1 is working correctly
"""
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'Model1_EmailContent'))

from phishing_detection_system import PhishingDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model1():
    """Test Model 1 loading and prediction"""
    print("="*60)
    print("Testing Model 1")
    print("="*60)
    
    # Check model path
    models_dir = project_root / 'trained_models' / 'Model1'
    print(f"\n1. Checking model path: {models_dir}")
    print(f"   Exists: {models_dir.exists()}")
    
    if not models_dir.exists():
        print("ERROR: Model 1 not found!")
        print(f"Expected path: {models_dir}")
        return False
    
    # List files in model directory
    print(f"\n2. Model directory contents:")
    for item in models_dir.iterdir():
        print(f"   - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    # Try to load model
    print(f"\n3. Loading Model 1...")
    try:
        detector = PhishingDetector(require_cuda=False)
        model_path_abs = str(models_dir.resolve())
        print(f"   Loading from: {model_path_abs}")
        detector.load_model(model_path_abs)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check if model object exists
    print(f"\n4. Verifying model object...")
    if detector.model is None:
        print("   ✗ ERROR: detector.model is None!")
        return False
    else:
        print(f"   ✓ Model object exists: {type(detector.model)}")
    
    # Check if predict method exists
    print(f"\n5. Checking predict method...")
    if not hasattr(detector, 'predict'):
        print("   ✗ ERROR: predict() method not found!")
        return False
    else:
        print("   ✓ predict() method exists")
    
    # Test prediction
    print(f"\n6. Testing prediction...")
    test_email = "Subject: Urgent Action Required\n\nYour account has been compromised. Click here immediately to verify: http://fake-bank.com/verify"
    
    try:
        result = detector.predict(test_email)
        print(f"   ✓ Prediction successful!")
        print(f"\n   Result:")
        print(f"   - Type: {type(result)}")
        print(f"   - Keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        print(f"   - is_phishing: {result.get('is_phishing', 'N/A')}")
        print(f"   - phishing_probability: {result.get('phishing_probability', 'N/A')}")
        print(f"   - legitimate_probability: {result.get('legitimate_probability', 'N/A')}")
        print(f"   - confidence: {result.get('confidence', 'N/A')}")
        
        # Validate result format
        required_keys = ['is_phishing', 'phishing_probability', 'legitimate_probability', 'confidence']
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            print(f"\n   ⚠ WARNING: Missing keys in result: {missing_keys}")
            return False
        
        # Check if probabilities are valid
        phishing_prob = result.get('phishing_probability', 0)
        legit_prob = result.get('legitimate_probability', 0)
        
        if phishing_prob == 0 and legit_prob == 1:
            print(f"\n   ⚠ WARNING: Result shows 0% phishing, 100% legitimate")
            print(f"   This might indicate the model is not working correctly")
        else:
            print(f"\n   ✓ Probabilities look valid")
        
        return True
        
    except Exception as e:
        print(f"   ✗ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model1()
    print("\n" + "="*60)
    if success:
        print("✓ Model 1 test PASSED")
    else:
        print("✗ Model 1 test FAILED")
    print("="*60)
    sys.exit(0 if success else 1)

