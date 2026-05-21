"""Patch the DCN C++/CUDA sources for torch 2.x compatibility.

Replacements:
  - `.type().is_cuda()`               -> `.is_cuda()`
  - `AT_DISPATCH_FLOATING_TYPES(x.type(), ...)`
                                      -> `AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), ...)`
  - `THCCeilDiv(a, b)`                -> `at::ceil_div(a, b)`

Re-runnable: each substitution is a no-op once applied.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "model" / "deformconv" / "src"
PATTERNS = [
    (re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.type\(\)\.is_cuda\(\)"), r"\1.is_cuda()"),
    (re.compile(r"AT_DISPATCH_FLOATING_TYPES\(\s*([A-Za-z_][A-Za-z0-9_]*)\.type\(\)"),
     r"AT_DISPATCH_FLOATING_TYPES(\1.scalar_type()"),
    (re.compile(r"THCCeilDiv\(([^,]+),\s*([^)]+)\)"), r"at::ceil_div(\1, \2)"),
]

changed = 0
for path in ROOT.rglob("*"):
    if path.suffix.lower() not in {".cu", ".cuh", ".cpp", ".h"}:
        continue
    text = path.read_text(encoding="utf-8")
    new = text
    for rx, repl in PATTERNS:
        new = rx.sub(repl, new)
    if new != text:
        path.write_text(new, encoding="utf-8")
        changed += 1
        print(f"patched {path.relative_to(ROOT.parent.parent.parent.parent.parent)}")

print(f"done; {changed} file(s) modified")
