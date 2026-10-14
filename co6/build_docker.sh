#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Building Docker Image for Diabetic Readmission Prediction"
echo "============================================================"
cd "$(dirname "$0")/.."
docker build -t diabetes-predict-app:1.0.0 .
echo "[SUCCESS] Docker image diabetes-predict-app:1.0.0 built successfully!"
