"""Render GitHub-ready SVG/PNG figures from a run summary, without model calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jrt.figures import render_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=ROOT / "data/items.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", help="Required when a summary contains multiple judges")
    parser.add_argument("--report", type=Path, help="Existing Markdown report to embed figures into")
    args = parser.parse_args()
    try:
        paths = render_figures(args.summary, args.items, args.output, args.judge, args.report)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"figure error: {exc}\n")
    print(f"Rendered {len(paths)} SVGs, 4 PNGs, and source/CSV files in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
