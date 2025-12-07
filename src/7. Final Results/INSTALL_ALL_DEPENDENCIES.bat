@echo off
REM Install All Dependencies for Portable PhishGuard
REM This installs PyTorch, TensorFlow, Transformers, and all ML dependencies

echo ========================================
echo PhishGuard - Installing All Dependencies
echo ========================================
echo.
echo This will install:
echo   - PyTorch (Model 1) - ~2.8 GB
echo   - TensorFlow (Model 2) - ~1.5 GB
echo   - Transformers and other ML libraries - ~1 GB
echo.
echo Total: ~5-6 GB download, 10-15 GB installed
echo Time: 20-40 minutes depending on internet speed
echo.
echo Press Ctrl+C to cancel, or
pause

cd /d "%~dp0"

REM Set all temp/cache directories to N: drive
set PIP_CACHE_DIR=N:\pip_cache
set TMPDIR=N:\temp
set TEMP=N:\temp
set TMP=N:\temp

REM Create directories if they don't exist
if not exist "N:\pip_cache" mkdir "N:\pip_cache"
if not exist "N:\temp" mkdir "N:\temp"

echo.
echo Using N: drive for all downloads and temp files
echo Cache: %PIP_CACHE_DIR%
echo Temp: %TMPDIR%
echo.

if not exist "python\python.exe" (
    echo ERROR: python\python.exe not found!
    echo Please extract Python embeddable package to: python\
    pause
    exit /b 1
)

echo ========================================
echo Step 1/4: Installing PyTorch (Model 1)
echo ========================================
echo This will take 10-15 minutes...
echo.
python\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-warn-script-location --cache-dir %PIP_CACHE_DIR%

if errorlevel 1 (
    echo.
    echo ERROR: PyTorch installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 2/4: Installing Transformers
echo ========================================
echo.
python\python.exe -m pip install transformers>=4.30.0 tokenizers>=0.13.0 --no-warn-script-location --cache-dir %PIP_CACHE_DIR%

if errorlevel 1 (
    echo.
    echo ERROR: Transformers installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 3/4: Installing TensorFlow (Model 2)
echo ========================================
echo This will take 5-10 minutes...
echo.
python\python.exe -m pip install tensorflow --no-warn-script-location --cache-dir %PIP_CACHE_DIR%

if errorlevel 1 (
    echo.
    echo ERROR: TensorFlow installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 4/4: Installing Other Dependencies
echo ========================================
echo.
python\python.exe -m pip install pandas>=1.5.0 numpy>=1.21.0 scikit-learn>=1.1.0 matplotlib>=3.5.0 seaborn>=0.11.0 tqdm>=4.64.0 accelerate>=0.20.0 datasets>=2.12.0 --no-warn-script-location --cache-dir %PIP_CACHE_DIR%

if errorlevel 1 (
    echo.
    echo ERROR: Dependencies installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo All dependencies are now installed in:
echo   python\Lib\site-packages\
echo.
echo Your project is now fully portable!
echo.
echo Next steps:
echo   1. Make sure trained_models\ folder exists
echo   2. Copy entire project to flash drive
echo   3. On another computer, just run: api_server\start_server.bat
echo.
pause

