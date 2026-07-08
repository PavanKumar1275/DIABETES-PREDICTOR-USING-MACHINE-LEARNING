@echo off
color 0A
title Diabetes Prediction System

echo.
echo =====================================================
echo    DIABETES PREDICTION SYSTEM                       
echo =====================================================
echo.

echo Starting Diabetes Prediction Application...
echo ========================================
echo.

echo Please wait while the application is starting...
echo.

REM Use the system Python installation to run the application
"C:\Users\Pavan Kumar S\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run diabetes_prediction_app.py

echo.
echo Application has been closed.
echo.

pause