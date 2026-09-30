#!/usr/bin/env python3
"""Rewrite runtime/vllm-overlay/SHA256SUMS.json after changing overlay files (install_overlay.py checks it)."""
import hashlib
import json
from pathlib import Path

OVERLAY = Path(__file__).resolve().parent.parent / "runtime" / "vllm-overlay"
MANIFEST = OVERLAY / "SHA256SUMS.json"
sums = {str(p.relative_to(OVERLAY)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OVERLAY.rglob("*")) if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts}
MANIFEST.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
print(f"{len(sums)} overlay files")
