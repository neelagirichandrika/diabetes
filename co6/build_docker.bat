@echo off
echo ============================================================
echo Building Docker Image for Diabetic Readmission Prediction
echo ============================================================
cd ..
docker build -t diabetes-predict-app:1.0.0 .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed. Ensure Docker Desktop is running.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Docker image diabetes-predict-app:1.0.0 built successfully!
pause
