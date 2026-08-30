"""Audit the item pool for confounds that would invalidate the experiment.

An item pool is not just "a bunch of questions". If the correct answer happens
to be longer than the wrong one across most items, then any length preference a
judge has will masquerade as competence, and every axis measurement is polluted.
This script is the gate: a pool that fails it cannot be used to make claims.

Usage:
    python scripts/validate_items.py data/items.jsonl
    python scripts/validate_items.py data/items.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

# Maximum tolerated relative length gap between correct and wrong candidates.
# Beyond this, a judge's length preference is no longer separable from its
# actual discrimination ability on that item.
MAX_REL_LEN_GAP = 0.15

# A pool this small cannot support the claims the preregistration makes.
MIN_ITEMS_FOR_CLAIMS = 150

MIN_ITEM_CHARS = 40
MAX_ITEM_CHARS = 1200


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check_schema(rows: list[dict]) -> list[str]:
    errs = []
    seen = set()
    for i, r in enumerate(rows, 1):
        for key in ("id", "domain", "question", "correct", "wrong"):
            if key not in r:
                errs.append(f"item #{i}: missing key {key!r}")
        rid = r.get("id")
        if rid in seen:
            errs.append(f"duplicate id {rid!r}")
        seen.add(rid)
        if not isinstance(r.get("correct"), str) or not isinstance(r.get("wrong"), str):
            errs.append(f"item {rid}: correct/wrong must be strings")
    return errs


def check_trivial(rows: list[dict]) -> list[str]:
    """Reject items where the two candidates are near-identical or degenerate."""
    errs = []
    for r in rows:
        c, w = _norm(r["correct"]), _norm(r["wrong"])
        if c == w:
            errs.append(f"item {r['id']}: correct and wrong are identical")
            continue
        if len(c) < MIN_ITEM_CHARS or len(w) < MIN_ITEM_CHARS:
            errs.append(
                f"item {r['id']}: candidate too short "
                f"(c={len(c)} w={len(w)}, need >= {MIN_ITEM_CHARS})"
            )
        if len(c) > MAX_ITEM_CHARS or len(w) > MAX_ITEM_CHARS:
            errs.append(
                f"item {r['id']}: candidate too long "
                f"(c={len(c)} w={len(w)}, cap {MAX_ITEM_CHARS})"
            )
        # Same content, different casing/punctuation => not a real discrimination task.
        if c.lower().rstrip(".!") == w.lower().rstrip(".!"):
            errs.append(f"item {r['id']}: candidates differ only by case/punctuation")
    return errs


def length_gap(rows: list[dict]) -> tuple[list[tuple[str, float]], float, float]:
    """Return per-item signed gap, mean signed gap, and share of items over cap."""
    gaps = []
    for r in rows:
        c, w = len(_norm(r["correct"])), len(_norm(r["wrong"]))
        gaps.append((r["id"], (c - w) / max(c, w, 1)))
    vals = [g for _, g in gaps]
    return gaps, statistics.mean(vals) if vals else 0.0, sum(
        1 for g in vals if abs(g) > MAX_REL_LEN_GAP
    ) / max(len(vals), 1)


def leakage_hints(rows: list[dict]) -> list[str]:
    """Surface cues that let a judge shortcut without reading the argument.

    These are warnings, not hard errors: some cues are unavoidable. But a pool
    where the correct answer is systematically more fluent, more hedged, or
    more structured makes 'substance' impossible to isolate.
    """
    warns = []

    def rate(pred) -> float:
        return sum(1 for r in rows if pred(r)) / max(len(rows), 1)

    hedge = re.compile(r"\b(however|whereas|note that|strictly|importantly|careful)\b", re.I)
    ch = rate(lambda r: bool(hedge.search(r["correct"])))
    wh = rate(lambda r: bool(hedge.search(r["wrong"])))
    if abs(ch - wh) > 0.25:
        warns.append(
            f"hedging-language imbalance: correct {ch:.0%} vs wrong {wh:.0%} "
            "(>25pp) - judges may key on tone"
        )

    # Report asymmetry, not raw share. Most items have the same digit count on
    # both sides, so "correct has more digits in only 5% of items" is balance,
    # not imbalance. What matters is whether one side wins far more often than
    # the other.
    cnum = rate(lambda r: len(re.findall(r"\d", r["correct"])) > len(re.findall(r"\d", r["wrong"])))
    wnum = rate(lambda r: len(re.findall(r"\d", r["wrong"])) > len(re.findall(r"\d", r["correct"])))
    if abs(cnum - wnum) > 0.30:
        lean = "correct" if cnum > wnum else "wrong"
        warns.append(
            f"numeric-density asymmetry: {lean} carries more digits far more often "
            f"({max(cnum, wnum):.0%} vs {min(cnum, wnum):.0%}) - amplifies the length confound"
        )

    clines = rate(lambda r: r["correct"].count("\n") > r["wrong"].count("\n"))
    wlines = rate(lambda r: r["wrong"].count("\n") > r["correct"].count("\n"))
    if abs(clines - wlines) > 0.30:
        lean = "correct" if clines > wlines else "wrong"
        warns.append(
            f"line-count asymmetry: {lean} has more line breaks more often "
            f"({max(clines, wlines):.0%} vs {min(clines, wlines):.0%}) - structural fluency cue"
        )

    admit = re.compile(r"\b(I'?m not sure|I don'?t know|unclear|cannot determine)\b", re.I)
    if rate(lambda r: bool(admit.search(r["correct"]))) > 0.05:
        warns.append("some correct answers express uncertainty - breaks the abstention axis baseline")
    return warns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--strict", action="store_true", help="also require the preregistered pool size")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"no such file: {args.path}")

    rows = load(args.path)
    print(f"Item pool: {args.path}  ({len(rows)} items)")
    print()

    domains = Counter(r.get("domain") for r in rows)
    print("Domains:")
    for d, n in sorted(domains.items()):
        print(f"  {d:<16} {n:>4}")
    print()

    errs = check_schema(rows) + check_trivial(rows)
    gaps, mean_gap, over = length_gap(rows)

    print("Length balance (signed gap = (|correct| - |wrong|) / max)")
    print(f"  mean signed gap        : {mean_gap:+.3f}   (target |.| <= 0.05)")
    print(f"  |gap| > {MAX_REL_LEN_GAP:<5}      : {over:.1%} of items   (target 0%)")
    worst = sorted(gaps, key=lambda x: -abs(x[1]))[:8]
    if worst:
        print("  worst offenders:")
        for rid, g in worst:
            print(f"    {rid:<8} {g:+.3f}")
    print()

    warns = leakage_hints(rows)
    if warns:
        print("Warnings:")
        for w in warns:
            print(f"  ! {w}")
        print()

    hard_fail = bool(errs)
    if abs(mean_gap) > 0.05:
        hard_fail = True
        errs.append(
            f"mean length gap {mean_gap:+.3f} exceeds 0.05: the pool systematically "
            "favours one candidate on length, which contaminates every axis"
        )
    if over > 0.10:
        hard_fail = True
        errs.append(
            f"{over:.0%} of items exceed the {MAX_REL_LEN_GAP:.0%} per-item length cap "
            "(tolerated: 10%)"
        )
    if args.strict and len(rows) < MIN_ITEMS_FOR_CLAIMS:
        hard_fail = True
        errs.append(
            f"pool has {len(rows)} items; preregistered claims need >= {MIN_ITEMS_FOR_CLAIMS} "
            f"(see scripts/power_analysis.py)"
        )

    if errs:
        print("FAIL:")
        for e in errs[:40]:
            print(f"  x {e}")
        if len(errs) > 40:
            print(f"  ... and {len(errs) - 40} more")
        return 1

    print("PASS: pool is balanced enough to support bias claims.")
    if not args.strict and len(rows) < MIN_ITEMS_FOR_CLAIMS:
        print(
            f"NOTE: {len(rows)} items is below the {MIN_ITEMS_FOR_CLAIMS} needed for "
            "small-effect claims. Findings are limited to h >= "
            f"{0.386 if len(rows) <= 40 else 0.2:.3f}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
