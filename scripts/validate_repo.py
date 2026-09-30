#!/usr/bin/env python3
"""GPU-free repository checks (CI): overlay manifest, pins, executables, no weights or credentials."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERLAY = ROOT / "runtime" / "vllm-overlay"
MANIFEST = OVERLAY / "SHA256SUMS.json"
EXECUTABLES = ("scripts/serve-container.sh", "scripts/docker_serve.sh", "scripts/preflight.sh", "scripts/bench.py",
               "scripts/validate_repo.py", "scripts/update_manifest.py", "tools/ram-balloon.sh")
# sha256 of lower-cased words that must not appear in public files (private host names and addresses)
PRIVATE = {
    "8d2384d8ba2ba606d97b3b2cc2021c0a458c8e857b6455ede8b2d8eabb0a5b98",
    "b420fb198da6609c0fb84d987da325c8aca016fd5edd7aaae2744e82301f75a7",
    "147f8ecdbce2270220a20e8d6ab03ab8b4f9b8e3fe22f2439d3d70fed1a4dbab",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    errors: list[str] = []
    expected = json.loads(MANIFEST.read_text())
    observed = {str(p.relative_to(OVERLAY)): sha256(p) for p in OVERLAY.rglob("*")
                if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts}
    if observed != expected:
        errors.append("overlay manifest mismatch (run scripts/update_manifest.py): "
                      f"missing={sorted(set(expected) - set(observed))} extra={sorted(set(observed) - set(expected))} "
                      f"changed={sorted(k for k in expected.keys() & observed.keys() if expected[k] != observed[k])}")

    lock = json.loads((ROOT / "repro.lock.json").read_text())
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    if f"FROM {lock['runtime']['base_image']}" not in dockerfile:
        errors.append("docker/Dockerfile base image differs from repro.lock.json")
    profile = (ROOT / lock["runtime"]["profile"]).read_text()
    for key, value in (("MAX_MODEL_LEN", lock["runtime"]["max_model_len"]),
                       ("QWEN38_HOT_ONLY", lock["runtime"]["hot_experts_per_layer"]),
                       ("KV_CACHE_DTYPE", lock["runtime"]["kv_cache_dtype"]),
                       ("MTP_DEPTH", lock["runtime"]["mtp_depth"])):
        if f'${{{key}:={value}}}' not in profile:
            errors.append(f"{lock['runtime']['profile']}: {key} default differs from repro.lock.json ({value})")
    revision = lock["checkpoint"]["revision"]
    if revision not in (ROOT / "README.md").read_text():
        errors.append("README.md does not pin the checkpoint revision from repro.lock.json")

    secret = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink: {rel}")
        if path.stat().st_size > 20 * 2**20:
            errors.append(f"file over 20 MiB: {rel}")
        if path.suffix.lower() in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}:
            errors.append(f"model payload must not be committed: {rel}")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if secret.search(text):
            errors.append(f"possible credential in {rel}")
        tokens = re.findall(r"[a-z0-9]+", text.lower()) + re.findall(r"[0-9]+(?:\.[0-9]+){3}", text)
        words = {hashlib.sha256(w.encode()).hexdigest() for w in tokens}
        if words & PRIVATE:
            errors.append(f"private host identifier in {rel}")
    for rel in EXECUTABLES:
        if not os.access(ROOT / rel, os.X_OK):
            errors.append(f"not executable: {rel}")

    if errors:
        print("repository validation failed:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"repository validation passed: {len(observed)} overlay files")


if __name__ == "__main__":
    main()
