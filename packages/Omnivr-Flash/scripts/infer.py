#!/usr/bin/env python3
"""Omnivr-Flash: single-step joint audio/video restoration."""

import os
from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parents[1]
PACKAGES = PACKAGE.parent
for source in [PACKAGES / "ltx-core/src", PACKAGE / "src"]:
    sys.path.insert(0, str(source))

# Set allocator/thread preferences before importing PyTorch. Explicit caller
# settings take precedence, including the older allocator environment variable.
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "8")

from omnivr_flash.cli import main


if __name__ == "__main__":
    main()
