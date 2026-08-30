"""Assemble the item pool from the per-domain source files.

Sources live in data/raw/<domain>.jsonl and are authored independently so each
domain can be audited and extended on its own. This script merges them,
interleaving domains so that item order carries no domain signal, and writes
the pool the runner consumes.

    python scripts/build_pool.py                 # writes data/items.jsonl
    python scripts/build_pool.py --out data/x.jsonl --shuffle 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DEFAULT_OUT = os.path.join(ROOT, "data", "items.jsonl")

DOMAIN_ORDER = ("arithmetic", "logic", "code", "fact", "constraint")


def read_domain(name: str) -> list[dict]:
    path = os.path.join(RAW_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing source file: {path}")
    rows = []
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: {exc}") from exc
    return rows


def interleave(by_domain: "OrderedDict[str, list[dict]]") -> list[dict]:
    """Round-robin across domains so no run of consecutive items shares a domain."""
    out: list[dict] = []
    queues = [list(v) for v in by_domain.values()]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--shuffle", type=int, default=0, help="seed for a final shuffling pass (0 keeps interleaved order)")
    ap.add_argument("--domains", nargs="*", default=list(DOMAIN_ORDER))
    args = ap.parse_args()

    by_domain: "OrderedDict[str, list[dict]]" = OrderedDict()
    for d in args.domains:
        rows = read_domain(d)
        for r in rows:
            r.setdefault("domain", d)
        by_domain[d] = rows
        print(f"  {d:<14} {len(rows):>4} items")

    pool = interleave(by_domain)
    if args.shuffle:
        random.Random(args.shuffle).shuffle(pool)

    seen: set[str] = set()
    for r in pool:
        if r["id"] in seen:
            raise SystemExit(f"duplicate item id {r['id']!r}")
        seen.add(r["id"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in pool:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    gaps = [
        (len(norm(r["correct"])) - len(norm(r["wrong"])))
        / max(len(norm(r["correct"])), len(norm(r["wrong"])), 1)
        for r in pool
    ]
    print()
    print(f"Wrote {len(pool)} items to {os.path.relpath(args.out, ROOT)}")
    print(f"  mean signed length gap: {sum(gaps) / len(gaps):+.4f}")
    print(f"  max |length gap|      : {max(abs(g) for g in gaps):.3f}")
    print()
    print("Next: python scripts/validate_items.py", os.path.relpath(args.out, ROOT), "--strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
