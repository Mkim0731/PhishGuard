#!/usr/bin/env python3
"""
Python Finder Utility
=====================
Finds Python executable on the system for portable deployment.
"""

import sys
import os
import subprocess
from pathlib import Path

def find_python():
    """Find Python executable"""
    # Try python command first
    try:
        result = subprocess.run(['python', '--version'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return 'python'
    except:
        pass
    
    # Try python3
    try:
        result = subprocess.run(['python3', '--version'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return 'python3'
    except:
        pass
    
    # Try py launcher (Windows)
    if sys.platform == 'win32':
        try:
            result = subprocess.run(['py', '--version'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return 'py'
        except:
            pass
    
    # Try common locations
    common_paths = []
    if sys.platform == 'win32':
        local_appdata = os.environ.get('LOCALAPPDATA', '')
        program_files = os.environ.get('ProgramFiles', '')
        program_files_x86 = os.environ.get('ProgramFiles(x86)', '')
        
        common_paths = [
            Path(local_appdata) / 'Programs' / 'Python',
            Path(program_files) / 'Python',
            Path(program_files_x86) / 'Python',
            Path('C:/Python'),
        ]
    else:
        common_paths = [
            Path('/usr/bin'),
            Path('/usr/local/bin'),
            Path.home() / '.local' / 'bin',
        ]
    
    for base_path in common_paths:
        if not base_path.exists():
            continue
        
        # Look for python.exe (Windows) or python (Unix)
        exe_name = 'python.exe' if sys.platform == 'win32' else 'python'
        
        # Check base path
        python_path = base_path / exe_name
        if python_path.exists():
            return str(python_path)
        
        # Check subdirectories (for versioned Python folders)
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                python_path = subdir / exe_name
                if python_path.exists():
                    return str(python_path)
    
    return None

if __name__ == '__main__':
    python_exe = find_python()
    if python_exe:
        print(f"Found Python: {python_exe}")
        sys.exit(0)
    else:
        print("ERROR: Python not found!")
        print("\nPlease install Python or add it to PATH")
        sys.exit(1)

