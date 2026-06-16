@echo off
echo ========================================
echo IMPROVED PIPELINE - Quick Run
echo ========================================
echo.
echo This will run the improved pipeline for both Cargo and Passengership
echo to achieve 80-85%+ accuracy.
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo [1/2] Running Cargo pipeline...
python project\run_improved_pipeline.py --ship_type Cargo --residual_infusion 0.45

echo.
echo [2/2] Running Passengership pipeline...
python project\run_improved_pipeline.py --ship_type Passengership --residual_infusion 0.45

echo.
echo ========================================
echo COMPLETE! Check project/outputs/improved_pipeline/ for results
echo ========================================
pause
