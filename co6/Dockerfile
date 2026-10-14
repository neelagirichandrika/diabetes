# ==============================================================================
# CO6 Dockerfile: Multi-Stage Production Build for Diabetic Readmission Prediction
# Curriculum Module: CO6 (ML Engineering — From Notebook to Production)
# ==============================================================================

FROM python:3.11-slim AS runtime

LABEL maintainer="Clinical ML Engineering Team" \
      module="CO6 Production Deployment" \
      description="Containerized Diabetic Inpatient Readmission Inference Service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_ENV=production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').getcode() == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--timeout", "60", "run_app:app"]
