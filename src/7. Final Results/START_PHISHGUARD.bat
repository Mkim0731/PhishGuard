@echo off
REM PhishGuard - Master Startup Script
REM This starts everything needed for PhishGuard to work

echo ========================================
echo PhishGuard - Starting System
echo ========================================
echo.

cd /d "%~dp0"

REM Check for Python
if not exist "python\python.exe" (
    echo ERROR: Python not found!
    echo.
    echo Please make sure python\ folder exists with Python embeddable package.
    echo See FLASH_DRIVE_SETUP.md for instructions.
    echo.
    pause
    exit /b 1
)

REM Check for models
if not exist "trained_models\Model1" (
    echo WARNING: Model 1 not found!
    echo.
    echo Models need to be trained first.
    echo Run: python\python.exe train_both_models.py
    echo.
    echo Continue anyway? (Y/N)
    set /p continue=
    if /i not "%continue%"=="Y" exit /b 1
)

if not exist "trained_models\Model2" (
    echo WARNING: Model 2 not found!
    echo.
    echo Models need to be trained first.
    echo Run: python\python.exe train_both_models.py
    echo.
    echo Continue anyway? (Y/N)
    set /p continue=
    if /i not "%continue%"=="Y" exit /b 1
)

echo.
echo Starting API server...
echo.
echo Server will run on: http://localhost:5000
echo.
echo Keep this window open!
echo.
echo To stop: Press Ctrl+C
echo.
echo ========================================
echo.

REM Start the API server
cd api_server
call start_server.bat

