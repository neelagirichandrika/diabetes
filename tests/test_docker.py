"""
Integration and configuration unit tests for CO6 Docker implementation.
Run with:
    py -m unittest tests/test_docker.py
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import app
from src.predict import SAMPLE_PATIENTS


class TestDockerImplementation(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_dockerfile_exists_and_security(self):
        dockerfile_path = os.path.join(PROJECT_ROOT, "Dockerfile")
        self.assertTrue(os.path.exists(dockerfile_path), "Root Dockerfile missing")

        with open(dockerfile_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("python:3.11-slim", content, "Expected official python:3.11-slim base image")
        self.assertIn("useradd", content, "Expected non-root user creation")
        self.assertIn("USER appuser", content, "Expected execution as non-root appuser")
        self.assertIn("EXPOSE 5000", content, "Expected port 5000 exposure")
        self.assertIn("HEALTHCHECK", content, "Expected HEALTHCHECK probe definition")
        self.assertIn("gunicorn", content, "Expected production WSGI server execution")

    def test_dockerignore_coverage(self):
        ignore_path = os.path.join(PROJECT_ROOT, ".dockerignore")
        self.assertTrue(os.path.exists(ignore_path), ".dockerignore missing")

        with open(ignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn(".venv", content, "Virtual env not excluded in .dockerignore")
        self.assertIn(".git", content, "Git metadata not excluded in .dockerignore")
        self.assertIn("__pycache__", content, "Bytecode cache not excluded in .dockerignore")
        self.assertIn("diabetic_data.csv", content, "Raw dataset not excluded from Docker context")

    def test_docker_compose_validity(self):
        compose_path = os.path.join(PROJECT_ROOT, "docker-compose.yml")
        self.assertTrue(os.path.exists(compose_path), "docker-compose.yml missing")

        with open(compose_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("5000:5000", content, "Port mapping 5000:5000 missing in compose")
        self.assertIn("healthcheck", content, "Healthcheck missing in compose")
        self.assertIn("restart: unless-stopped", content, "Restart policy missing in compose")

    def test_container_healthcheck_endpoint(self):
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("pipeline_available"))
        self.assertTrue(data.get("container_ready"))

    def test_api_docker_status_endpoint(self):
        response = self.app.get("/api/docker-status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ready")
        self.assertTrue(data.get("dockerfile_available"))
        self.assertTrue(data.get("docker_compose_available"))
        self.assertTrue(data.get("dockerignore_available"))
        self.assertTrue(data.get("co6_module_available"))

    def test_co6_report_and_architecture_generated(self):
        report_path = os.path.join(PROJECT_ROOT, "co6", "co6_docker_report.txt")
        diagram_path = os.path.join(PROJECT_ROOT, "co6", "docker_container_architecture.png")

        self.assertTrue(os.path.exists(report_path), "co6/co6_docker_report.txt missing")
        self.assertTrue(os.path.exists(diagram_path), "co6/docker_container_architecture.png missing")

        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        self.assertIn("CERTIFIED PRODUCTION READY", report_text)


if __name__ == "__main__":
    unittest.main()
