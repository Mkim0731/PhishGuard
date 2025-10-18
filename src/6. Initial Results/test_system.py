#!/usr/bin/env python3
"""
Test script for Phishing Detection System
==========================================

This script provides a quick test of the phishing detection system
to ensure all components work correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ torch imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA not available - system will exit")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        print("✓ sklearn imported successfully")
    except ImportError as e:
        print(f"✗ sklearn import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test if all required data files exist"""
    print("\nTesting data files...")
    
    required_files = [
        'CEAS_08.csv',
        'Enron.csv', 
        'Ling.csv',
        'Nazario.csv',
        'Nigerian_Fraud.csv',
        'SpamAssasin.csv'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist

def test_main_script():
    """Test if the main script can be imported"""
    print("\nTesting main script...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Import the main module
        from phishing_detection_system import PhishingDetector
        print("✓ PhishingDetector class imported successfully")
        
        # Test initialization
        detector = PhishingDetector()
        print("✓ PhishingDetector initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Main script test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("PHISHING DETECTION SYSTEM - TEST SUITE")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
        print("✓ Import test PASSED")
    else:
        print("✗ Import test FAILED")
    
    # Test data files
    if test_data_files():
        tests_passed += 1
        print("✓ Data files test PASSED")
    else:
        print("✗ Data files test FAILED")
    
    # Test main script
    if test_main_script():
        tests_passed += 1
        print("✓ Main script test PASSED")
    else:
        print("✗ Main script test FAILED")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready to run.")
        print("\nTo run the full system:")
        print("python phishing_detection_system.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
