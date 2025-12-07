@echo off
echo ========================================
echo PhishGuard Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or 3.9 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/4] Python found: 
python --version
echo.

echo [2/4] Installing Model 1 dependencies...
cd Model1_EmailContent
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Model 1 dependencies
    pause
    exit /b 1
)
cd ..

echo.
echo [3/4] Installing Model 2 dependencies...
cd Model2_URLDetection\Source_Code
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Model 2 dependencies
    pause
    exit /b 1
)
cd ..\..

echo.
echo [4/4] Installing API server dependencies...
cd api_server
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install API server dependencies
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Train the models (if not already trained):
echo    python train_both_models.py
echo.
echo 2. Start the API server:
echo    start_server.bat
echo.
echo 3. Load the Chrome extension in Chrome
echo.
pause






