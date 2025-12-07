# Bundling Python with PhishGuard

You can include Python with your project so users don't need to install it separately.

## Option 1: Python Embeddable Package (Recommended)

Python.org provides an "embeddable" version that can be bundled with your project.

### Steps:

1. **Download Python Embeddable Package:**
   - Go to: https://www.python.org/downloads/windows/
   - Scroll down to "Windows embeddable package"
   - Download Python 3.11 or 3.12 (64-bit recommended)
   - Extract to: `PhishGuard-main/src/Final_Project/python/`

2. **Install pip in embeddable Python:**
   - Download `get-pip.py`: https://bootstrap.pypa.io/get-pip.py
   - Place it in the `python/` folder
   - Run: `python/python.exe -m ensurepip --upgrade`
   - Or: `python/python.exe get-pip.py`

3. **Install dependencies:**
   ```bash
   python/python.exe -m pip install -r api_server/requirements.txt
   ```

4. **Update startup scripts** to use bundled Python (see below)

### Folder Structure:
```
PhishGuard-main/
└── src/
    └── Final_Project/
        ├── python/              (embeddable Python)
        │   ├── python.exe
        │   ├── python311.dll
        │   └── ...
        ├── api_server/
        ├── chrome_extension/
        └── ...
```

## Option 2: Portable Python Distribution

Use a portable Python distribution like **WinPython** or **PortablePython**.

### WinPython:
1. Download from: https://winpython.github.io/
2. Extract to: `PhishGuard-main/src/Final_Project/python/`
3. Install dependencies using bundled Python
4. Update startup scripts

## Option 3: PyInstaller (Standalone Executable)

Create a standalone executable for the API server (more complex).

### Pros:
- Single executable file
- No Python installation needed

### Cons:
- Large file size (~200-500 MB)
- Slower startup time
- More complex to maintain
- May have issues with TensorFlow/PyTorch

## Recommended: Option 1 (Embeddable Python)

This is the best balance of:
- ✅ Small size (~20-30 MB)
- ✅ Easy to bundle
- ✅ Works with all dependencies
- ✅ Portable

## Updating Startup Scripts

After bundling Python, update `start_server.bat` and `start_server.ps1` to check for bundled Python first:

```batch
REM Check for bundled Python first
if exist "%~dp0..\python\python.exe" (
    set PYTHON_EXE=%~dp0..\python\python.exe
    goto :found
)

REM Then try system Python...
```

See `start_server_bundled.bat` for complete example.

