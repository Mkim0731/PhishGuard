# PhishGuard Training Launcher
# This script will find Python and start training

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PhishGuard - Finding Python and Starting Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Try to find Python
$pythonPaths = @(
    "python",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe",
    "$env:ProgramFiles\Python*\python.exe",
    "$env:ProgramFiles(x86)\Python*\python.exe",
    "C:\Python*\python.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python3*\python.exe"
)

$pythonExe = $null

foreach ($path in $pythonPaths) {
    try {
        if ($path -eq "python" -or $path -eq "py") {
            $result = Get-Command $path -ErrorAction SilentlyContinue
            if ($result) {
                $pythonExe = $result.Source
                Write-Host "Found Python: $pythonExe" -ForegroundColor Green
                break
            }
        } else {
            $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found -and (Test-Path $found.FullName)) {
                $pythonExe = $found.FullName
                Write-Host "Found Python: $pythonExe" -ForegroundColor Green
                break
            }
        }
    } catch {
        continue
    }
}

if (-not $pythonExe) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure Python 3.14 is installed and:" -ForegroundColor Yellow
    Write-Host "1. Restart your PowerShell/terminal" -ForegroundColor Yellow
    Write-Host "2. Or add Python to PATH manually" -ForegroundColor Yellow
    Write-Host "3. Or run Python using full path" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To find Python, check:" -ForegroundColor Yellow
    Write-Host "  - $env:LOCALAPPDATA\Programs\Python" -ForegroundColor Gray
    Write-Host "  - $env:ProgramFiles\Python*" -ForegroundColor Gray
    Write-Host ""
    pause
    exit 1
}

# Verify Python version
Write-Host "Checking Python version..." -ForegroundColor Cyan
$version = & $pythonExe --version 2>&1
Write-Host "Python version: $version" -ForegroundColor Green
Write-Host ""

# Change to project directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Run training
& $pythonExe train_both_models.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training script finished." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
pause






