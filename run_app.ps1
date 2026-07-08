# PowerShell script to run the Diabetes Prediction Application
Write-Host "====================================================="
Write-Host "   DIABETES PREDICTION SYSTEM                       "
Write-Host "====================================================="
Write-Host ""

Write-Host "Starting Diabetes Prediction Application..."
Write-Host "========================================"
Write-Host ""

Write-Host "Please wait while the application is starting..."
Write-Host ""

# Use the system Python installation to run the application
& "C:\Users\Pavan Kumar S\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run diabetes_prediction_app.py

Write-Host ""
Write-Host "Application has been closed."
Write-Host ""

Write-Host "Press any key to continue..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")