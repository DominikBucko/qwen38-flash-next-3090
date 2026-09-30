#!/usr/bin/env python3
"""Render docs/images/results.svg from benchmarks/2026-09-30/summary.json (standard library only)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "benchmarks" / "2026-09-30" / "summary.json"
OUT = ROOT / "docs" / "images" / "results.svg"
W, H = 960, 470
BLUE, ORANGE, INK, MUTED, GRID = "#2563eb", "#ea7a24", "#111827", "#6b7280", "#e5e7eb"


def label(res: dict) -> str:
    return f"{res['prompt_tokens']:,} + {res['completion_tokens']:,}"


def panel(x0: float, title: str, unit: str, rows: list[tuple[str, float]], vmax: float, color: str) -> list[str]:
    out = [f'<text x="{x0}" y="114" font-size="17" font-weight="700" fill="{INK}">{title}</text>',
           f'<text x="{x0}" y="134" font-size="13" fill="{MUTED}">{unit}</text>']
    bar_x, bar_w = x0 + 128, 290
    for step in range(5):
        gx = bar_x + bar_w * step / 4
        out.append(f'<line x1="{gx:.1f}" y1="150" x2="{gx:.1f}" y2="{150 + 62 * len(rows) - 14}" stroke="{GRID}"/>')
        out.append(f'<text x="{gx:.1f}" y="{150 + 62 * len(rows) + 4}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="middle">{vmax * step / 4:,.0f}</text>')
    for i, (name, value) in enumerate(rows):
        y = 156 + 62 * i
        w = bar_w * value / vmax
        out.append(f'<text x="{x0}" y="{y + 24}" font-size="13.5" fill="{INK}">{name}</text>')
        out.append(f'<rect x="{bar_x}" y="{y}" width="{w:.1f}" height="36" rx="5" fill="{color}"/>')
        out.append(f'<text x="{bar_x + w + 8:.1f}" y="{y + 24}" font-size="15" font-weight="700" '
                   f'fill="{INK}">{value:,.0f}</text>' if value >= 100 else
                   f'<text x="{bar_x + w + 8:.1f}" y="{y + 24}" font-size="15" font-weight="700" '
                   f'fill="{INK}">{value:.1f}</text>')
    return out


def render(summary: dict) -> str:
    run = next(r for r in summary["runs"] if r["id"] == summary["headline_run"])
    order = sorted(run["results"], key=lambda r: -r["prompt_tokens"])
    first = min(run["results"], key=lambda r: r["prompt_tokens"])
    rows_p = [(label(r) + (" *" if r is first else ""), r["prefill_tok_s"]) for r in order]
    rows_d = [(label(r) + (" *" if r is first else ""), r["decode_tok_s"]) for r in order]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" rx="14" fill="#ffffff"/>',
        f'<text x="32" y="46" font-size="22" font-weight="800" fill="{INK}">Qwen3.8-Flash-Next on one RTX 3090 '
        f'(24 GB) + 64 GB RAM</text>',
        f'<text x="32" y="72" font-size="14" fill="{MUTED}">{summary["subtitle"]}</text>',
    ]
    parts += panel(32, "Prefill", "input tokens / time to first token (tok/s)", rows_p, 2500, BLUE)
    parts += panel(500, "Decode", "output tokens / s after the first token", rows_d, 60, ORANGE)
    parts.append(f'<text x="32" y="{H - 18}" font-size="12" fill="{MUTED}">* first request after the server '
                 f'start. Requests are input + output tokens, run one at a time. Source: '
                 f'benchmarks/2026-09-30/summary.json</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(json.loads(SUMMARY.read_text())))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
