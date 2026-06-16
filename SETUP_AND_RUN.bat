@echo off
echo =====================================================
echo SETUP AND RUN - Improved Pipeline
echo =====================================================
echo.
echo This script will:
echo 1. Install dependencies
echo 2. Verify previous pipeline has run
echo 3. Run improved pipeline for both ships
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo [Step 1/4] Installing dependencies...
pip install -r project\requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [Step 2/4] Checking for previous residuals...
if not exist "project\outputs\decomposition\separated_audio\Cargo" (
    echo.
    echo WARNING: No previous residuals found for Cargo
    echo Running original pipeline first to generate residuals...
    python project\post_run.py
    if %errorlevel% neq 0 (
        echo ERROR: Original pipeline failed
        pause
        exit /b 1
    )
)

echo.
echo [Step 3/4] Running improved pipeline for Cargo...
python project\run_improved_pipeline.py --ship_type Cargo --residual_infusion 0.45
if %errorlevel% neq 0 (
    echo ERROR: Cargo pipeline failed
    pause
    exit /b 1
)

echo.
echo [Step 4/4] Running improved pipeline for Passengership...
python project\run_improved_pipeline.py --ship_type Passengership --residual_infusion 0.45
if %errorlevel% neq 0 (
    echo ERROR: Passengership pipeline failed
    pause
    exit /b 1
)

echo.
echo =====================================================
echo SUCCESS! Check results:
echo project\outputs\improved_pipeline\improvement_report_Cargo.md
echo project\outputs\improved_pipeline\improvement_report_Passengership.md
echo =====================================================
pause
