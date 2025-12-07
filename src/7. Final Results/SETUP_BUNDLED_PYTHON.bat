@echo off
REM Setup script for bundling Python with PhishGuard
REM This will download and set up embeddable Python

echo ========================================
echo PhishGuard - Bundle Python Setup
echo ========================================
echo.
echo This script will help you bundle Python with PhishGuard
echo so users don't need to install Python separately.
echo.
echo IMPORTANT: You need to manually download Python embeddable package first!
echo.
echo Steps:
echo 1. Go to: https://www.python.org/downloads/windows/
echo 2. Scroll to "Windows embeddable package"
echo 3. Download Python 3.11 or 3.12 (64-bit)
echo 4. Extract to: Final_Project\python\
echo 5. Run this script again
echo.
pause

REM Check if python folder exists
if not exist "python\" (
    echo.
    echo ERROR: python\ folder not found!
    echo.
    echo Please extract Python embeddable package to: python\
    echo.
    pause
    exit /b 1
)

if not exist "python\python.exe" (
    echo.
    echo ERROR: python\python.exe not found!
    echo.
    echo Please extract Python embeddable package to: python\
    echo.
    pause
    exit /b 1
)

echo.
echo Found Python embeddable package!
echo.

REM Install pip
echo Installing pip...
python\python.exe -m ensurepip --upgrade
if errorlevel 1 (
    echo.
    echo Attempting to install pip using get-pip.py...
    echo Downloading get-pip.py...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'python\get-pip.py'"
    python\python.exe python\get-pip.py
)

echo.
echo Installing dependencies...
python\python.exe -m pip install --upgrade pip
python\python.exe -m pip install -r api_server\requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Python is now bundled with your project!
echo.
echo To use bundled Python:
echo - Replace start_server.bat with start_server_bundled.bat
echo - Replace start_server.ps1 with start_server_bundled.ps1
echo.
echo Or manually update the scripts to check for bundled Python first.
echo.
pause

