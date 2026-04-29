@echo off
title Meme Sentiment Analyzer
echo ============================================
echo   Multimodal Meme Sentiment Analysis
echo ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

:: Create virtual environment if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ============================================
echo   Choose an option:
echo ============================================
echo   1. Download dataset from Kaggle
echo   2. Train model (Phase 3 - recommended)
echo   3. Train model (Phase 1 - overfit demo)
echo   4. Train model (Phase 2 - improved)
echo   5. Launch web demo
echo   6. Predict from image
echo ============================================
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    python train.py --download --phase 3
) else if "%choice%"=="2" (
    python train.py --phase 3
) else if "%choice%"=="3" (
    python train.py --phase 1
) else if "%choice%"=="4" (
    python train.py --phase 2
) else if "%choice%"=="5" (
    echo.
    echo Starting web server at http://localhost:5000
    python webapp\app.py
) else if "%choice%"=="6" (
    set /p img_path="Enter image path: "
    python predict.py --image "%img_path%"
) else (
    echo Invalid choice!
)

pause
