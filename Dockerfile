# ==============================================================================
# Dockerfile: Diabetic Inpatient Readmission Prediction Service
# Curriculum: Module 6 / CO6 (ML Engineering & Production Deployment)
# ==============================================================================

# Stage 1: Runtime Base Image
FROM python:3.11-slim AS runtime

# Metadata labels
LABEL maintainer="Clinical ML Engineering Team" \
      module="CO6 - ML Engineering & Production Containerization" \
      version="1.0.0" \
      description="Containerized clinical prediction web application and REST API"

# Set deterministic Python and environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_ENV=production

# Set application working directory
WORKDIR /app

# Install minimal OS dependencies and clean up cache in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Security Hardening: Run as dedicated non-root user (Principle of Least Privilege / HIPAA)
RUN useradd -m -u 1000 appuser

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application modules, models, static assets, templates, and documentation
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY docs/ docs/
COPY eda/ eda/
COPY cleaning/ cleaning/
COPY encoding/ encoding/
COPY scaling/ scaling/
COPY split/ split/
COPY splitting_criteria/ splitting_criteria/
COPY logistic_regression/ logistic_regression/
COPY decision_tree/ decision_tree/
COPY regularization/ regularization/
COPY bias_variance/ bias_variance/
COPY coefficients/ coefficients/
COPY model_comparison/ model_comparison/
COPY pipeline/ pipeline/
COPY co6/ co6/
COPY run_app.py .

# Enforce non-root ownership on working directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose internal web server port
EXPOSE 5000

# Automated Container Healthcheck verifying /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').getcode() == 200 else 1)"

# Production Entrypoint: Run with Gunicorn WSGI Server (2 workers, 2 threads, 60s timeout)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "run_app:app"]
