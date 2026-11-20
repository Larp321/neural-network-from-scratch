"""Lets the examples run straight from a clone (``python examples/spiral.py``) without installing."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
IMAGES = os.path.join(ROOT, "docs", "images")
os.makedirs(IMAGES, exist_ok=True)
