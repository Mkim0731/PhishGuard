# Portable Setup Guide

This guide helps you run PhishGuard from a flash drive or any location on any computer.

## ✅ What's Already Portable

- **Python Code**: All Python scripts use relative paths
- **Model Paths**: Models are loaded from `trained_models/` folder (relative)
- **Dataset Paths**: Datasets are loaded from current directory (relative)
- **Chrome Extension**: Uses relative paths

## 🔧 What Needs Configuration

### 1. Python Executable

The startup scripts (`start_server.bat`, `start_server.ps1`) now **auto-detect Python**, but if Python is not in PATH, you may need to configure it.

**Option A: Add Python to PATH (Recommended)**
- Install Python and check "Add Python to PATH" during installation
- Or manually add Python folder to Windows PATH

**Option B: Edit Startup Scripts**
- Open `api_server/start_server.bat` or `start_server.ps1`
- Find the line that sets `PYTHON_EXE` or `$pythonExe`
- Change it to your Python path, e.g.:
  - `set PYTHON_EXE=C:\Python39\python.exe`
  - `$pythonExe = "C:\Python39\python.exe"`

## 📁 Project Structure (Portable)

```
PhishGuard-main/
└── src/
    └── Final_Project/
        ├── Model1_EmailContent/        (relative paths)
        ├── Model2_URLDetection/        (relative paths)
        ├── trained_models/             (relative paths)
        ├── api_server/                 (relative paths)
        ├── chrome_extension/          (relative paths)
        └── train_both_models.py        (uses relative paths)
```

**All paths are relative to the project root!**

## 🚀 Running on a New Computer

### Step 1: Copy Project
- Copy entire `PhishGuard-main` folder to flash drive or new computer
- Keep the folder structure intact

### Step 2: Install Python (if not installed)
- Download Python from https://www.python.org/downloads/
- **Important**: Check "Add Python to PATH" during installation

### Step 3: Install Dependencies
```bash
cd PhishGuard-main/src/Final_Project
python -m pip install -r api_server/requirements.txt
```

### Step 4: Start API Server
- **Windows**: Double-click `api_server/start_server.bat`
- **PowerShell**: Right-click `api_server/start_server.ps1` → Run with PowerShell

### Step 5: Load Chrome Extension
- Go to `chrome://extensions/`
- Enable Developer mode
- Click "Load unpacked"
- Select `chrome_extension` folder

## ⚙️ Configuration File (Optional)

If you want to specify Python path manually, create `config.txt` in the project root:

```
PYTHON_PATH=C:\Python39\python.exe
```

Then modify startup scripts to read from this file.

## 🔍 Verifying Portability

All Python code uses:
- `Path(__file__).parent` - Gets script directory
- Relative paths like `'trained_models/Model1'` - Relative to project root
- No hardcoded absolute paths (except in startup scripts, which now auto-detect)

## 📝 Notes

- **Models must be trained first** - Copy `trained_models/` folder if already trained
- **Python must be installed** - The project doesn't include Python
- **Dependencies must be installed** - Run `pip install -r requirements.txt`
- **Chrome Extension** - Works from any location (uses relative paths)

