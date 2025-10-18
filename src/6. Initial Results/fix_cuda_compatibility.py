#!/usr/bin/env python3
"""
CUDA Compatibility Fix for RTX 5060 Ti
======================================

This script fixes CUDA compatibility issues with RTX 5060 Ti
by installing the correct PyTorch version and setting up environment variables.
"""

import subprocess
import sys
import os
import torch

def fix_cuda_compatibility():
    """Fix CUDA compatibility issues"""
    print("Fixing CUDA compatibility for RTX 5060 Ti...")
    
    # Set environment variables for CUDA compatibility
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("✓ Environment variables set")
    
    # Check current PyTorch version
    try:
        print(f"Current PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Compute Capability: {props.major}.{props.minor}")
            
            # Test CUDA functionality
            try:
                test_tensor = torch.randn(10, 10).cuda()
                result = torch.mm(test_tensor, test_tensor.t())
                print("✓ CUDA functionality test passed")
                return True
            except Exception as e:
                print(f"✗ CUDA functionality test failed: {e}")
                return False
        else:
            print("✗ CUDA not available")
            return False
            
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def reinstall_pytorch():
    """Reinstall PyTorch with correct CUDA support"""
    print("\nReinstalling PyTorch with CUDA support...")
    
    # Uninstall current PyTorch
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'], 
                      check=True, capture_output=True)
        print("✓ Uninstalled current PyTorch")
    except:
        print("⚠️  Could not uninstall PyTorch (may not be installed)")
    
    # Install PyTorch with CUDA 11.8 support
    try:
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch>=2.0.0+cu118',
            'torchvision>=0.15.0+cu118', 
            'torchaudio>=2.0.0+cu118',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ]
        
        subprocess.run(cmd, check=True)
        print("✓ Installed PyTorch with CUDA 11.8 support")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("CUDA COMPATIBILITY FIX FOR RTX 5060 Ti")
    print("=" * 60)
    
    # Test current setup
    if fix_cuda_compatibility():
        print("\n🎉 CUDA compatibility is working!")
        print("You can now run the phishing detection system.")
    else:
        print("\n⚠️  CUDA compatibility issues detected.")
        print("Attempting to fix by reinstalling PyTorch...")
        
        if reinstall_pytorch():
            print("\n🔄 PyTorch reinstalled. Testing again...")
            if fix_cuda_compatibility():
                print("\n🎉 CUDA compatibility fixed!")
                print("You can now run the phishing detection system.")
            else:
                print("\n❌ CUDA compatibility still not working.")
                print("Please check:")
                print("1. NVIDIA drivers are up to date")
                print("2. CUDA toolkit is installed")
                print("3. GPU is properly connected")
        else:
            print("\n❌ Failed to reinstall PyTorch.")
            print("Please install manually:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
