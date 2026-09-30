#!/usr/bin/env python3
"""Quick functional check of a running server: a chat answer with reasoning, and a tool call (standard library only).

  python3 scripts/smoke_test.py [--base-url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def chat(base: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read()), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    failures = 0

    reply, dt = chat(base, {"model": args.model, "temperature": 0, "max_tokens": 2048, "messages": [
        {"role": "user", "content": "What is 17 * 23? Reply with the number only."}]})
    msg = reply["choices"][0]["message"]
    answer = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    ok = "391" in answer
    failures += not ok
    print(f"[{'ok' if ok else 'FAIL'}] arithmetic: answer {answer!r}, {len(reasoning)} reasoning chars, {dt:.1f} s")

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    reply, dt = chat(base, {"model": args.model, "temperature": 0, "max_tokens": 2048, "tools": tools,
                            "messages": [{"role": "user", "content": "What's the weather in Bratislava right now?"}]})
    calls = reply["choices"][0]["message"].get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and \
        "bratislava" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
    failures += not ok
    print(f"[{'ok' if ok else 'FAIL'}] tool call: {[(c['function']['name'], c['function']['arguments']) for c in calls]}, "
          f"{dt:.1f} s")

    reply, dt = chat(base, {"model": args.model, "temperature": 0, "max_tokens": 1024, "messages": [
        {"role": "user", "content": "Write a Python function is_prime(n) and nothing else."}],
        "chat_template_kwargs": {"enable_thinking": False}})
    code = reply["choices"][0]["message"].get("content") or ""
    ok = "def is_prime" in code
    failures += not ok
    usage = reply.get("usage", {})
    print(f"[{'ok' if ok else 'FAIL'}] code, thinking off: {usage.get('completion_tokens')} tokens in {dt:.1f} s")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
