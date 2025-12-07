@echo off
REM Fix embeddable Python configuration to enable pip

echo ========================================
echo Fixing Embeddable Python Configuration
echo ========================================
echo.

cd /d "%~dp0"

if not exist "python\python.exe" (
    echo ERROR: python\python.exe not found!
    echo Please extract Python embeddable package to: python\
    pause
    exit /b 1
)

echo Found Python embeddable package!
echo.

REM Find the _pth file
for %%F in (python\python*.pth) do (
    set PTH_FILE=%%F
    goto :found_pth
)

echo ERROR: python*.pth file not found!
pause
exit /b 1

:found_pth
echo Found configuration file: %PTH_FILE%
echo.

REM Backup original
copy "%PTH_FILE%" "%PTH_FILE%.backup" >nul
echo Created backup: %PTH_FILE%.backup
echo.

REM Read and modify the file
echo Updating configuration...
powershell -Command "$content = Get-Content '%PTH_FILE%' -Raw; $content = $content -replace '^import site', '#import site'; if ($content -notmatch 'Scripts') { $content += \"`r`nScripts`r`n\"; }; Set-Content -Path '%PTH_FILE%' -Value $content -NoNewline"

if errorlevel 1 (
    echo ERROR: Failed to update configuration file
    pause
    exit /b 1
)

echo OK: Configuration updated!
echo.

REM Test pip
echo Testing pip...
python\python.exe -m pip --version

if errorlevel 1 (
    echo.
    echo WARNING: pip still not working
    echo Trying to reinstall pip...
    echo.
    python\python.exe -m ensurepip --upgrade
    if errorlevel 1 (
        echo.
        echo Downloading get-pip.py...
        powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'python\get-pip.py'"
        python\python.exe python\get-pip.py
    )
)

echo.
echo ========================================
echo Configuration complete!
echo ========================================
echo.
echo Now you can install dependencies:
echo   python\python.exe -m pip install -r api_server\requirements.txt
echo.
pause

