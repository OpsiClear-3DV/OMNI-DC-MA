"""Shared pytest bootstrap.

This codebase (inherited from princeton-vl/OMNI-DC) resolves its modules as a
flat namespace with the working directory at ``src/`` — e.g. ``from config
import args``, ``from backbone import Backbone``. Rather than rewrite ~30
interconnected upstream files to a package layout (high risk against a
verified-working inference path, zero functional gain), tests reproduce that
import environment here, once, instead of each test file re-doing it.
"""

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
_PATHS = (SRC, SRC / "model", SRC / "model" / "deformconv")


@pytest.fixture(autouse=True)
def _omnidc_import_env(monkeypatch):
    for p in _PATHS:
        monkeypatch.syspath_prepend(str(p))
    monkeypatch.chdir(SRC)
    # demo.py / config.py read sys.argv at import time; give a benign default
    # so pytest's own file paths and switches are not parsed as model args.
    monkeypatch.setattr(sys, "argv", ["pytest", "--gpus", "0"])
    yield
    os.chdir(Path(__file__).resolve().parents[1])
