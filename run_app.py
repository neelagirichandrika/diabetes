"""
Root launcher for Diabetic Inpatient Readmission Prediction Web Application.

To run from PyCharm:
    Right-click 'run_app.py' and select "Run 'run_app'"
Or via terminal:
    py run_app.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")

    print("=" * 60)
    print("DIABETIC INPATIENT READMISSION PREDICTION — WEB SERVER")
    print("=" * 60)
    print(f"Starting server on: http://{host}:{port}")
    print("Container & Local execution ready.")
    print("Press Ctrl+C to stop the server.\n")
    app.run(host=host, port=port, debug=debug)
