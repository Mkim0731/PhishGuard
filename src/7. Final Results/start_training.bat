@echo off
echo ============================================================
echo PhishGuard - Starting Model Training
echo ============================================================
echo.
echo This will train both models with progress tracking.
echo Estimated time: ~2.25 hours total
echo.
echo Progress and time estimates will be shown during training.
echo.
pause

cd /d "%~dp0"
python train_both_models.py

pause






