"""Check that your environment is ready. Run it after following docs/07-setup.md.

    python scripts/check_setup.py

Each line prints OK or a problem with what to do about it.
"""

import importlib
import socket
import sys

PACKAGES = [
    "poke_env",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "lightgbm",
    "matplotlib",
    "datasets",
    "pokeagent",
]


def check_python() -> bool:
    ok = sys.version_info >= (3, 10)
    print(
        f"{'OK ' if ok else 'BAD'} Python {sys.version.split()[0]}"
        + ("" if ok else "  -> install Python 3.11 or 3.12")
    )
    return ok


def check_packages() -> bool:
    all_ok = True
    for name in PACKAGES:
        try:
            importlib.import_module(name)
            print(f"OK  import {name}")
        except ImportError as e:
            all_ok = False
            print(f'BAD import {name}: {e}  -> run: pip install -e ".[dev]" inside your venv')
    return all_ok


def check_server(host: str = "localhost", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"OK  Showdown server reachable at {host}:{port}")
            return True
    except OSError:
        print(
            f"BAD no Showdown server at {host}:{port}  -> in the pokemon-showdown folder run:"
            " node pokemon-showdown start --no-security"
        )
        return False


if __name__ == "__main__":
    results = [check_python(), check_packages(), check_server()]
    print(
        "\nAll good. Next: python scripts/play_baselines.py --games 20"
        if all(results)
        else "\nFix the BAD lines above, then run this again."
    )
    sys.exit(0 if all(results) else 1)
