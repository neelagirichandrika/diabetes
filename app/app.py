"""
Flask Web Application for Diabetic Inpatient Readmission Prediction.
Provides a comprehensive clinical ML web interface:
  - Inpatient Risk Prediction Form & Real Inference
  - Exploratory Data Analysis (EDA) & Visualizations
  - Preprocessing, Scaling & Encoding Progression
  - Candidate Model Comparison & Diagnostics (Regularization, Bias-Variance, Feature Weights)
  - REST API endpoint
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import METRICS_PATH, PIPELINE_PATH
from src.predict import predict_patient, SAMPLE_PATIENTS

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load model evaluation metadata if available
model_metadata = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)
    except Exception as e:
        print(f"Notice: Could not load metadata: {e}")


def read_report(module_dir, filename):
    """Safely reads report text lines from module directories."""
    path = os.path.join(PROJECT_ROOT, module_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def read_curriculum_guide(filename):
    """Safely reads curriculum module markdown from docs/curriculum/."""
    path = os.path.join(PROJECT_ROOT, "docs", "curriculum", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            prediction_result = predict_patient(form_data)
        except Exception as e:
            prediction_result = {
                "error": True,
                "summary": f"Prediction error: {str(e)}"
            }

    # Load actual module reports
    reports = {
        "ingestion": read_report("ingestion", "ingestion_report.txt"),
        "eda": read_report("eda", "eda_summary_report.txt"),
        "cleaning": read_report("cleaning", "cleaning_report.txt"),
        "encoding": read_report("encoding", "encoding_analysis_report.txt"),
        "scaling": read_report("scaling", "scaling_report.txt"),
        "split": read_report("split", "split_report.txt"),
        "splitting_criteria": read_report("splitting_criteria", "splitting_criteria_report.txt"),
        "logistic": read_report("logistic_regression", "logistic_report.txt"),
        "decision_tree": read_report("decision_tree", "decision_tree_report.txt"),
        "regularization": read_report("regularization", "regularization_report.txt"),
        "bias_variance": read_report("bias_variance", "bias_variance_report.txt"),
        "coefficients": read_report("coefficients", "coefficients_report.txt"),
        "model_comparison": read_report("model_comparison", "model_comparison_report.txt"),
        "pipeline": read_report("pipeline", "pipeline_verification_report.txt"),
        "co1": read_report("co1", "co1_lifecycle_report.txt"),
        "co2": read_report("co2", "co2_linear_models_report.txt"),
        "co3": read_report("co3", "co3_tree_models_report.txt"),
        "co4": read_report("co4", "co4_unsupervised_report.txt"),
        "co5": read_report("co5", "co5_evaluation_report.txt"),
        "co6": read_report("co6", "co6_docker_report.txt")
    }

    return render_template(
        "index.html",
        result=prediction_result,
        form_data=form_data,
        metadata=model_metadata,
        sample_patients=SAMPLE_PATIENTS,
        reports=reports
    )


@app.route("/artifacts/<module>/<path:filename>")
def serve_artifact(module, filename):
    """Serves plots directly from their respective workflow folders."""
    safe_modules = [
        "eda", "cleaning", "encoding", "scaling", "split", "splitting_criteria",
        "logistic_regression", "decision_tree", "regularization", "bias_variance",
        "coefficients", "model_comparison", "models",
        "co1", "co2", "co3", "co4", "co5", "co6"
    ]
    if module not in safe_modules:
        return "Not found", 404
    directory = os.path.join(PROJECT_ROOT, module)
    return send_from_directory(directory, filename)


@app.route("/health", methods=["GET"])
def health():
    """Container health check endpoint for Docker HEALTHCHECK and orchestration probes."""
    pipeline_loaded = os.path.exists(PIPELINE_PATH)
    status_code = 200 if pipeline_loaded else 503
    return jsonify({
        "status": "healthy" if pipeline_loaded else "unhealthy",
        "pipeline_available": pipeline_loaded,
        "service": "diabetic-readmission-prediction",
        "container_ready": True,
        "version": "1.0.0"
    }), status_code


@app.route("/api/docker-status", methods=["GET"])
def api_docker_status():
    """Returns status and configuration of CO6 Docker containerization."""
    dockerfile_exists = os.path.exists(os.path.join(PROJECT_ROOT, "Dockerfile"))
    compose_exists = os.path.exists(os.path.join(PROJECT_ROOT, "docker-compose.yml"))
    ignore_exists = os.path.exists(os.path.join(PROJECT_ROOT, ".dockerignore"))
    co6_report_exists = os.path.exists(os.path.join(PROJECT_ROOT, "co6", "co6_docker_report.txt"))

    return jsonify({
        "status": "ready" if (dockerfile_exists and compose_exists) else "incomplete",
        "dockerfile_available": dockerfile_exists,
        "docker_compose_available": compose_exists,
        "dockerignore_available": ignore_exists,
        "co6_module_available": os.path.exists(os.path.join(PROJECT_ROOT, "co6")),
        "co6_report_available": co6_report_exists,
        "exposed_port": int(os.environ.get("PORT", 5000)),
        "host": os.environ.get("HOST", "127.0.0.1"),
        "model_loaded": os.path.exists(PIPELINE_PATH)
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic inference."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400
        result = predict_patient(data)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Returns trained model metrics and architecture summary."""
    return jsonify(model_metadata)


@app.route("/api/curriculum/<int:module_id>", methods=["GET"])
def api_curriculum(module_id):
    """Returns raw markdown and status for a specified curriculum module (1 to 6)."""
    filename_map = {
        1: "Module_01_ML_Lifecycle_DiabetesPredict.md",
        2: "Module_02_Linear_Models_at_Depth.md",
        3: "Module_03_Tree_Based_Models.md",
        4: "Module_04_Unsupervised_Learning.md",
        5: "Module_05_Evaluation_Selection_Calibration.md",
        6: "Module_06_ML_Engineering_Production.md"
    }
    fname = filename_map.get(module_id)
    if not fname:
        return jsonify({"status": "error", "message": "Invalid module ID. Expected 1 through 6."}), 404
    content = read_curriculum_guide(fname)
    return jsonify({
        "status": "success",
        "module_id": module_id,
        "filename": fname,
        "content": content
    })


@app.route("/api/co-status", methods=["GET"])
def api_co_status():
    """Returns audit and readiness status for all 6 Course Outcomes (CO1 to CO6)."""
    co_modules = {
        "co1": {
            "name": "CO1: ML Lifecycle, Ingestion, Hygiene & Preprocessing",
            "runner": "co1/run_co1_lifecycle.py",
            "report_available": os.path.exists(os.path.join(PROJECT_ROOT, "co1", "co1_lifecycle_report.txt")),
            "visual_available": os.path.exists(os.path.join(PROJECT_ROOT, "co1", "co1_lifecycle_architecture.png")),
            "status": "ready"
        },
        "co2": {
            "name": "CO2: Linear Models, Regularization & Odds Ratios",
            "runner": "co2/run_co2_linear_models.py",
            "report_available": os.path.exists(os.path.join(PROJECT_ROOT, "co2", "co2_linear_models_report.txt")),
            "visual_available": os.path.exists(os.path.join(PROJECT_ROOT, "co2", "co2_odds_ratios_regularization.png")),
            "status": "ready"
        },
        "co3": {
            "name": "CO3: Tree-Based Models, Splitting & Ensembles",
            "runner": "co3/run_co3_tree_models.py",
            "report_available": os.path.exists(os.path.join(PROJECT_ROOT, "co3", "co3_tree_models_report.txt")),
            "visual_available": os.path.exists(os.path.join(PROJECT_ROOT, "co3", "co3_tree_depth_overfitting.png")),
            "status": "ready"
        },
        "co4": {
            "name": "CO4: Unsupervised Learning, K-Means & PCA Phenotypes",
            "runner": "co4/run_co4_unsupervised.py",
            "report_available": os.path.exists(os.path.join(PROJECT_ROOT, "co4", "co4_unsupervised_report.txt")),
            "visual_available": os.path.exists(os.path.join(PROJECT_ROOT, "co4", "patient_clusters_pca.png")),
            "status": "ready"
        },
        "co5": {
            "name": "CO5: Model Evaluation, Selection & Calibration",
            "runner": "co5/run_co5_evaluation.py",
            "report_available": os.path.exists(os.path.join(PROJECT_ROOT, "co5", "co5_evaluation_report.txt")),
            "visual_available": os.path.exists(os.path.join(PROJECT_ROOT, "co5", "co5_evaluation_calibration.png")),
            "status": "ready"
        },
        "co6": {
            "name": "CO6: ML Engineering, Docker Containerization & Deployment",
            "runner": "co6/run_co6_docker.py",
            "report_available": os.path.exists(os.path.join(PROJECT_ROOT, "co6", "co6_docker_report.txt")),
            "visual_available": os.path.exists(os.path.join(PROJECT_ROOT, "co6", "docker_container_architecture.png")),
            "status": "ready"
        }
    }
    return jsonify({
        "status": "success",
        "total_cos": 6,
        "certified_count": sum(1 for c in co_modules.values() if c["report_available"] and c["visual_available"]),
        "course_outcomes": co_modules
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print("\nStarting Diabetes Prediction Application...")
    print(f"Accessible at: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
