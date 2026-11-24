"""python -m pokeagent.dashboard [trace.jsonl]: start the dashboard in the browser."""

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")

if __name__ == "__main__":
    command = [sys.executable, "-m", "streamlit", "run", str(APP), "--", *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))
