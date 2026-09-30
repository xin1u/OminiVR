#!/usr/bin/env python3
"""Compatibility entry point for the original multistep OmniVR commands."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[1] / "infer_multistep.py"
    runpy.run_path(str(entry), run_name="__main__")
