"""Inline the viz JSON into the template -> a single self-contained HTML file.

The output `viz/surface_code.html` has no external dependencies (no server, no
network, no CDN) — open it directly in a browser.

Run (after build_viz_data.py):
    python scripts/build_viz_html.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "viz" / "surface_code.template.html"
DATA = ROOT / "data" / "viz" / "surface_code_viz.json"
OUT = ROOT / "viz" / "surface_code.html"


def main() -> int:
    tpl = TEMPLATE.read_text()
    data = json.loads(DATA.read_text())
    payload = json.dumps(data, separators=(",", ":"))
    if "/*__DATA__*/{}" not in tpl:
        raise SystemExit("placeholder /*__DATA__*/{} not found in template")
    html = tpl.replace("/*__DATA__*/{}", payload)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
