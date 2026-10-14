#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Running Diabetic Readmission Prediction in Docker Container"
echo "============================================================"
cd "$(dirname "$0")/.."
echo "Starting container on http://localhost:5000 via Docker Compose..."
docker compose up --build
