@echo off
title PhishGuard API Server
color 0A

echo ========================================
echo PhishGuard API Server
echo ========================================
echo.
echo Starting server on http://localhost:5000
echo.
echo Keep this window open while using the extension
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

cd /d "%~dp0"
cd api_server
python app.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Server failed to start
    echo ========================================
    echo.
    echo Possible issues:
    echo 1. Models not trained - Run: python train_both_models.py
    echo 2. Dependencies not installed - Run: setup.bat
    echo 3. Port 5000 already in use
    echo.
    pause
)






