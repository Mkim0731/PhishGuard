#!/usr/bin/env python3
"""
GPU Setup Script for Phishing Detection System
==============================================

This script helps set up CUDA support for the phishing detection system.
It checks for CUDA availability and provides installation instructions.
"""

import subprocess
import sys
import platform

def check_cuda_installation():
    """Check if CUDA is properly installed"""
    print("Checking CUDA installation...")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print(result.stdout.split('\n')[2])  # GPU info line
        else:
            print("✗ NVIDIA GPU not detected or nvidia-smi not found")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found. Please install NVIDIA drivers.")
        return False
    
    try:
        # Check CUDA version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA toolkit installed")
            print(result.stdout.split('\n')[3])  # CUDA version line
        else:
            print("✗ CUDA toolkit not found")
            return False
    except FileNotFoundError:
        print("✗ nvcc not found. Please install CUDA toolkit.")
        return False
    
    return True

def check_pytorch_cuda():
    """Check if PyTorch has CUDA support"""
    print("\nChecking PyTorch CUDA support...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            print(f"✓ Current GPU: {torch.cuda.get_device_name()}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA not available in PyTorch")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def install_cuda_pytorch():
    """Install PyTorch with CUDA support"""
    print("\nInstalling PyTorch with CUDA support...")
    
    # Determine CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if 'release 11.8' in result.stdout:
            cuda_version = 'cu118'
        elif 'release 11.7' in result.stdout:
            cuda_version = 'cu117'
        elif 'release 12.1' in result.stdout:
            cuda_version = 'cu121'
        else:
            cuda_version = 'cu118'  # Default to CUDA 11.8
    except:
        cuda_version = 'cu118'  # Default to CUDA 11.8
    
    print(f"Installing PyTorch with CUDA {cuda_version} support...")
    
    # Install PyTorch with CUDA
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        f'torch>=2.0.0+{cuda_version}',
        f'torchvision>=0.15.0+{cuda_version}',
        f'torchaudio>=2.0.0+{cuda_version}',
        '--index-url', f'https://download.pytorch.org/whl/{cuda_version}'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ PyTorch with CUDA support installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch with CUDA: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("GPU SETUP FOR PHISHING DETECTION SYSTEM")
    print("=" * 60)
    
    # Check system info
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    
    # Check CUDA installation
    cuda_installed = check_cuda_installation()
    
    # Check PyTorch CUDA support
    pytorch_cuda = check_pytorch_cuda()
    
    print("\n" + "=" * 60)
    
    if cuda_installed and pytorch_cuda:
        print("🎉 CUDA setup is complete! You can now run the phishing detection system.")
        print("\nTo run the system:")
        print("python phishing_detection_system.py")
    elif cuda_installed and not pytorch_cuda:
        print("⚠️  CUDA is installed but PyTorch doesn't have CUDA support.")
        print("Installing PyTorch with CUDA support...")
        
        if install_cuda_pytorch():
            print("\n🎉 PyTorch with CUDA support installed! You can now run the system.")
        else:
            print("\n❌ Failed to install PyTorch with CUDA support.")
            print("Please install manually:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("❌ CUDA setup incomplete. Please install:")
        print("\n1. NVIDIA GPU drivers")
        print("2. CUDA toolkit (version 11.8 or 12.1)")
        print("3. PyTorch with CUDA support")
        print("\nInstallation commands:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
