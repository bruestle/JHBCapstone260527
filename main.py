from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_entry = Path(__file__).resolve().parent / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_entry), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
