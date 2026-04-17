from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
