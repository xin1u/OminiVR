#!/usr/bin/env python3
"""Repository-level entry point for multistep OmniVR restoration."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    entry = Path(__file__).resolve().parent / "packages/omini-restore-trainer/scripts/infer.py"
    runpy.run_path(str(entry), run_name="__main__")
