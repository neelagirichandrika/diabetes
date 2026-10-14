@echo off
echo ============================================================
echo Running Diabetic Readmission Prediction in Docker Container
echo ============================================================
cd ..
echo Starting container on http://localhost:5000 via Docker Compose...
docker compose up --build
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container. Ensure Docker Desktop is running.
    pause
    exit /b %errorlevel%
)
pause
