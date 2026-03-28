"""
conftest.py — pytest configuration.

Adds src/ and scripts/ to sys.path so tests can import from both
without needing the editable install to be active in every environment.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
