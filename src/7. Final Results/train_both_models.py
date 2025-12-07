#!/usr/bin/env python3
"""
Master Training Script
======================

Trains both models (Model 1: Email Content, Model 2: URL Detection)
and saves them to trained_models/ folder.

This script should be run ONCE to train both models.
After training, models can be reused without retraining.
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_model_exists(model_path):
    """Check if model is already trained"""
    if model_path.name == "Model1":
        # Check for DistilBERT model files
        required_files = ['config.json', 'pytorch_model.bin']
        return all((model_path / f).exists() for f in required_files)
    elif model_path.name == "Model2":
        # Check for CNN model files
        return (model_path / 'model.h5').exists() and (model_path / 'tokenizer.pkl').exists()
    return False

def train_model1():
    """Train Model 1 (Email Content - DistilBERT)"""
    print_header("TRAINING MODEL 1: Email Content Detection (DistilBERT)")
    
    # Check for GPU BEFORE anything else
    try:
        import torch
        if not torch.cuda.is_available():
            print("="*70)
            print("ERROR: CUDA GPU IS REQUIRED FOR MODEL 1 TRAINING!")
            print("="*70)
            print("Training on CPU would take DAYS to complete.")
            print("The script will exit if GPU is not available.")
            print("="*70)
            return False
        else:
            print(f"OK: GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"OK: CUDA available - training will proceed\n")
    except ImportError:
        print("ERROR: PyTorch not installed. Cannot check for GPU.")
        return False
    
    model_path = Path(__file__).parent / 'trained_models' / 'Model1'
    
    # Check if already trained
    if check_model_exists(model_path):
        print(f"OK: Model 1 already trained! Found at: {model_path}")
        response = input("Do you want to retrain? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping Model 1 training...\n")
            return True
    
    print("WARNING: Model 1 training requires CUDA GPU and takes ~45 minutes")
    print("Starting training...\n")
    
    # Change to Model1 directory
    model1_dir = Path(__file__).parent / 'Model1_EmailContent'
    script_path = model1_dir / 'phishing_detection_system.py'
    
    if not script_path.exists():
        print(f"❌ Error: Training script not found at {script_path}")
        return False
    
    # Run training
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(model1_dir),
            check=True
        )
        print("\nOK: Model 1 training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Model 1 training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False

def train_model2():
    """Train Model 2 (URL Detection - CNN)"""
    print_header("TRAINING MODEL 2: URL Phishing Detection (CNN)")
    
    # Check for GPU BEFORE anything else
    try:
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices('GPU')
        if not gpu_devices:
            print("="*70)
            print("WARNING: NO GPU DETECTED FOR MODEL 2!")
            print("="*70)
            print("Training on CPU would take 10+ hours (30 epochs).")
            print("The training script will ask for confirmation.")
            print("="*70 + "\n")
        else:
            print(f"OK: GPU detected: {len(gpu_devices)} device(s)")
            for i, gpu in enumerate(gpu_devices):
                print(f"   GPU {i}: {gpu.name}")
            print("OK: GPU will be used for training\n")
    except ImportError:
        print("WARNING: TensorFlow not installed. Cannot check for GPU.")
    
    model_path = Path(__file__).parent / 'trained_models' / 'Model2'
    
    # Check if already trained
    if check_model_exists(model_path):
        print(f"OK: Model 2 already trained! Found at: {model_path}")
        response = input("Do you want to retrain? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping Model 2 training...\n")
            return True
    
    print("WARNING: Model 2 training takes ~1.5 hours (30 epochs)")
    print("Starting training...\n")
    
    # Change to Model2 directory
    model2_dir = Path(__file__).parent / 'Model2_URLDetection' / 'Source_Code'
    script_path = model2_dir / 'reproduction.py'
    
    if not script_path.exists():
        print(f"❌ Error: Training script not found at {script_path}")
        return False
    
    # Run training
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(model2_dir),
            check=True
        )
        print("\nOK: Model 2 training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Model 2 training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False

def verify_models():
    """Verify both models are trained and saved"""
    print_header("VERIFYING TRAINED MODELS")
    
    base_path = Path(__file__).parent / 'trained_models'
    model1_path = base_path / 'Model1'
    model2_path = base_path / 'Model2'
    
    model1_ok = check_model_exists(model1_path)
    model2_ok = check_model_exists(model2_path)
    
    if model1_ok:
        print("OK: Model 1 (Email Content) - Trained and saved")
    else:
        print("ERROR: Model 1 (Email Content) - Not found or incomplete")
    
    if model2_ok:
        print("OK: Model 2 (URL Detection) - Trained and saved")
    else:
        print("ERROR: Model 2 (URL Detection) - Not found or incomplete")
    
    return model1_ok and model2_ok

def main():
    """Main training function"""
    start_time = datetime.now()
    
    print_header("PHISHGUARD DUAL-MODEL TRAINING")
    print("This script will train both models for phishing detection.")
    print("Training is a one-time process - models will be saved for reuse.\n")
    
    # Verify directory structure
    base_path = Path(__file__).parent
    if not base_path.exists():
        print(f"ERROR: Project directory not found: {base_path}")
        sys.exit(1)
    
    # Create trained_models directory
    trained_models_dir = base_path / 'trained_models'
    trained_models_dir.mkdir(exist_ok=True)
    (trained_models_dir / 'Model1').mkdir(exist_ok=True)
    (trained_models_dir / 'Model2').mkdir(exist_ok=True)
    
    # Ask user which models to train
    print("Which models would you like to train?")
    print("1. Both models (Model 1 + Model 2)")
    print("2. Model 1 only (Email Content)")
    print("3. Model 2 only (URL Detection)")
    print("4. Skip training (verify existing models)")
    
    # Check for command line argument, otherwise prompt
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        print(f"\nUsing command line argument: {choice}")
    else:
        choice = input("\nEnter choice (1-4): ").strip()
    
    results = {'model1': False, 'model2': False}
    
    if choice == '1' or choice == '2':
        results['model1'] = train_model1()
    
    if choice == '1' or choice == '3':
        results['model2'] = train_model2()
    
    if choice == '4':
        pass  # Skip training, just verify
    
    # Verify models
    all_trained = verify_models()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("TRAINING SUMMARY")
    
    if all_trained:
        print("🎉 SUCCESS! Both models are trained and ready to use.")
        print("\nNext steps:")
        print("1. Models are saved in trained_models/ folder")
        print("2. You can now use prediction methods")
        print("3. Proceed to create API server and Chrome extension")
    else:
        print("WARNING: Some models are missing or incomplete.")
        print("Please train the missing models before proceeding.")
    
    print(f"\nTotal time: {duration}")
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING: Training interrupted by user")
        sys.exit(1)

