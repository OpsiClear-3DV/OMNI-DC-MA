#!/usr/bin/env python
"""Repo-root launcher for OMNI-DC-MA inference.

The codebase uses bare sibling imports (``from config import args``) and
expects the working directory to be ``src/``. Rather than restructure every
upstream module, this shim makes the demo runnable from anywhere:

    uv run python run_demo.py --demo_rgb a.jpg --demo_depth a.npy --demo_out_dir out/

It chdirs into ``src/`` and hands argv straight to ``demo.py``.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

if __name__ == "__main__":
    os.chdir(SRC)
    sys.path.insert(0, str(SRC))
    # demo.py reads sys.argv via config.py's argparse; pass everything through.
    runpy.run_path(str(SRC / "demo.py"), run_name="__main__")
