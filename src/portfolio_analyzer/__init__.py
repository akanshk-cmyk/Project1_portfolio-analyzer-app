"""Convenience entry point for the uv-managed local application."""

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli

    entry = Path(__file__).resolve().parents[2] / "app.py"
    sys.argv = ["streamlit", "run", str(entry), *sys.argv[1:]]
    cli.main()
