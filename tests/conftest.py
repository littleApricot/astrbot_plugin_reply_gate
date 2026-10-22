"""Load the plugin from its parent directory for standalone verification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
