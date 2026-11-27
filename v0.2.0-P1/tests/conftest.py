"""Make backend/ importable for the test suite."""
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

# round-30: server.py opens a daily log file at import time. Point it at a
# scratch dir so a test run never litters the repo, and pinned before any
# test module imports server.
os.environ.setdefault("AIGC_LOG_DIR", str(Path(tempfile.gettempdir()) / "aigc-test-logs"))
