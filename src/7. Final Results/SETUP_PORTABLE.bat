@echo off
REM PhishGuard Portable Setup Script
REM This script helps set up PhishGuard on a new computer

echo ========================================
echo PhishGuard - Portable Setup
echo ========================================
echo.
echo This script will help you set up PhishGuard on this computer.
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is installed
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% == 0 (
    python --version
    echo Python found!
    goto :check_deps
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    py --version
    echo Python found!
    goto :check_deps
)

echo.
echo ERROR: Python not found!
echo.
echo Please install Python from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:check_deps
echo.
echo Checking dependencies...
echo.

REM Check if requirements.txt exists
if not exist "api_server\requirements.txt" (
    echo ERROR: requirements.txt not found!
    echo Make sure you're running this from the Final_Project folder.
    pause
    exit /b 1
)

echo Installing dependencies...
echo This may take a few minutes...
echo.

python -m pip install --upgrade pip
python -m pip install -r api_server\requirements.txt

if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo Setup complete!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. If models are not trained, run: python train_both_models.py
    echo 2. Start the API server: double-click api_server\start_server.bat
    echo 3. Load the Chrome extension from chrome_extension folder
    echo.
) else (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check the error messages above.
    echo.
)

pause

