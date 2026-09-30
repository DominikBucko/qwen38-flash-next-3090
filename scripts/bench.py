#!/usr/bin/env python3
"""Measure prefill and decode speed of a running server, one request at a time (standard library only).

Each request is a code-review prompt of exactly N tokens built from this repository's runtime sources (tokenized
by the server), then M greedy output tokens. A unique cache_salt keeps the prefix cache out of the measurement.

  python3 scripts/bench.py                                  # 4K+256, 32K+512, 131K+512, 8K+1024
  python3 scripts/bench.py --shapes 8192:1024 --out results/8k.json

Prefill tok/s = prompt tokens / time to first token. Decode tok/s = (output tokens - 1) / (total - first token).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = (
    "Review this code for correctness bugs. For each bug, explain the failure scenario and propose a fix.",
    "Explain how the request lifecycle works across these modules, then list the three riskiest invariants.",
    "Refactor the most complex function you find into smaller functions; show the new code.",
    "Write unit tests for the cache logic in this code, covering edge cases.",
)
METRICS = ("num_preemptions_total", "spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total",
           "prompt_tokens_total", "generation_tokens_total")


def post(base: str, path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def metrics(base: str) -> dict:
    try:
        text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    except OSError:
        return {}
    out = {}
    for name in METRICS:
        values = re.findall(rf"^vllm:{name}(?:\{{[^}}]*\}})? ([0-9.e+]+)$", text, re.M)
        if values:
            out[name] = sum(float(v) for v in values)
    return out


def corpus(corpus_dir: Path) -> str:
    files = sorted(p for p in corpus_dir.rglob("*") if p.suffix in {".py", ".cpp", ".cu"} and p.is_file())
    if not files:
        raise SystemExit(f"no source files under {corpus_dir}")
    return "".join(f"\n# ===== {p.relative_to(corpus_dir)} =====\n" + p.read_text(errors="replace") for p in files)


def prompt_of(base: str, model: str, ids: list[int], n: int, offset: int, task: str) -> str:
    window = [ids[(offset + i) % len(ids)] for i in range(n)]
    text = post(base, "/detokenize", {"model": model, "tokens": window})["prompt"]
    return f"<|im_start|>user\n{text}\n\n{task}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def run_one(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    before = metrics(base)
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}, "cache_salt": f"bench-{time.time_ns()}"}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = usage = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if ttft is None and data.get("choices") and data["choices"][0].get("text"):
                ttft = time.perf_counter() - t0
            if data.get("usage"):
                usage = data["usage"]
    total = time.perf_counter() - t0
    time.sleep(1)
    after = metrics(base)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    prompt_tokens = usage["prompt_tokens"] if usage else None
    completion = usage["completion_tokens"] if usage else None
    drafts = delta.get("spec_decode_num_drafts_total") or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if ttft and prompt_tokens else None,
        "decode_tok_s": round((completion - 1) / (total - ttft), 2) if ttft and completion and total > ttft else None,
        "preemptions": delta.get("num_preemptions_total"),
        "accepted_per_draft": round(delta["spec_decode_num_accepted_tokens_total"] / drafts, 3) if drafts else None,
    }


def host_info() -> dict:
    info = {"python": platform.python_version(), "kernel": platform.release()}
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
        info["cpu"] = cpu
        info["mem_total_gib"] = round(next(int(line.split()[1]) for line in open("/proc/meminfo")
                                           if line.startswith("MemTotal:")) / 2**20, 1)
    except (OSError, StopIteration):
        pass
    try:
        gpu = os.environ.get("GPU_DEVICE", "0")
        info["gpu"] = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--shapes", default="4096:256,32768:512,131072:512,8192:1024",
                    help="comma-separated INPUT:OUTPUT token counts, run in order")
    ap.add_argument("--corpus", type=Path, default=ROOT / "runtime" / "vllm-overlay")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    ids = post(base, "/tokenize", {"model": args.model, "prompt": corpus(args.corpus),
                                   "add_special_tokens": False})["tokens"]
    results = []
    print(f"{'input':>8} {'output':>6} {'first token':>11} {'prefill tok/s':>13} {'decode tok/s':>12}", flush=True)
    for i, shape in enumerate(args.shapes.split(",")):
        n, m = (int(v) for v in shape.split(":"))
        prompt = prompt_of(base, args.model, ids, n, 5000 + 997 * i, TASKS[i % len(TASKS)])
        res = {"requested_input": n, "requested_output": m, **run_one(base, args.model, prompt, m)}
        results.append(res)
        print(f"{res['prompt_tokens']:>8} {res['completion_tokens']:>6} {res['ttft_s']:>10.1f}s "
              f"{res['prefill_tok_s']:>13,.0f} {res['decode_tok_s']:>12.1f}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"host": host_info(), "base_url": base, "results": results}, indent=1) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
