"""Pytest configuration."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Always run in mock mode for tests
os.environ["FRANKA_MOCK"] = "1"
os.environ["CAMERA_MOCK"] = "1"
