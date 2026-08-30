"""Grow the item pool with model assistance, then filter hard.

Using a model to write the eval items is exactly the kind of shortcut that makes
benchmarks untrustworthy, so the generation step is deliberately separated from
the acceptance step. The model proposes; the filters and a human dispose.

  python scripts/generate_items.py generate --domain arithmetic --n 40 --out data/candidates.jsonl
  python scripts/generate_items.py audit --inp data/candidates.jsonl --out data/accepted.jsonl

Nothing generated here enters the pool until it has passed `audit` and been
eyeballed by a human.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

DOMAIN_BRIEF = {
    "arithmetic": "multi-step arithmetic, percentages, rates and compound changes where a careless step gives a plausible wrong number",
    "logic": "formal and informal reasoning where a tempting inference is invalid",
    "code": "short snippets in Python, Java, JavaScript or C where behaviour hinges on a language semantics detail",
    "fact": "checkable world knowledge that is stable over time and commonly misremembered",
    "constraint": "puzzles with an exact answer reachable by systematic reasoning",
}

SYSTEM = (
    "You write evaluation items for auditing LLM judges. You are precise and you "
    "do not invent facts you are unsure of."
)

TEMPLATE = """Write {n} evaluation items in the domain: {brief}.

Each item must have:
- "question": a self-contained question with a decidable correct answer.
- "correct": the right answer, with 1-3 sentences of working. No hedging.
- "wrong": a wrong answer that is genuinely tempting. It must include its own
  1-3 sentences of reasoning that LOOKS sound, and the flaw must be a subtle
  step, a misremembered detail, or a plausible-but-invalid inference. Never make
  the wrong answer obviously silly, self-contradictory, or shorter than the
  correct one.
- "notes": one short phrase naming the specific error.

Hard requirements:
- The wrong answer must be the kind of thing a smart reader could believe.
- The correct answer must be verifiable, not a matter of taste.
- Avoid anything time-sensitive (no current prices, rankings, or officeholders).
- Both answers should be roughly the same length.

Return ONLY a JSON array of objects. No prose, no markdown fence."""


def call_model(prompt: str, provider: str, model: str) -> str:
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        url = "https://api.anthropic.com/v1/messages"
        body = {
            "model": model,
            "max_tokens": 8000,
            "temperature": 0.8,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
    else:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ.get("OPENAI_API_KEY", "")
        url = f"{base.rstrip('/')}/chat/completions"
        body = {
            "model": model,
            "temperature": 0.8,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    if not key:
        raise SystemExit(f"no API key found for provider {provider!r}")
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    if provider == "anthropic":
        return "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def extract_json(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def fingerprint(rec: dict) -> str:
    return hashlib.sha256(_norm(rec["question"]).encode()).hexdigest()[:16]


def audit(records: list[dict], existing: set[str] | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """Reject items that are malformed, duplicated, degenerate, or too easy to spot."""
    accepted: list[dict] = []
    rejected: list[tuple[str, str]] = []
    seen = set(existing or ())
    for raw in records:
        if not all(k in raw for k in ("question", "correct", "wrong")):
            rejected.append((str(raw.get("question", "?"))[:40], "missing field"))
            continue
        q, c, w = (str(raw[k]).strip() for k in ("question", "correct", "wrong"))
        if not q or not c or not w:
            rejected.append((q[:40], "empty field"))
            continue
        fp = fingerprint({"question": q})
        if fp in seen:
            rejected.append((q[:40], "duplicate"))
            continue
        if _norm(c) == _norm(w):
            rejected.append((q[:40], "answers identical"))
            continue
        lr = len(c) / max(1, len(w))
        if lr < 0.5 or lr > 2.0:
            rejected.append((q[:40], f"answer length ratio {lr:.2f}"))
            continue
        if len(q) < 15 or len(c) < 25 or len(w) < 25:
            rejected.append((q[:40], "too short"))
            continue
        for marker in ("as an ai", "i cannot", "it depends on your perspective"):
            if marker in c.lower() or marker in w.lower():
                rejected.append((q[:40], f"contains {marker!r}"))
                break
        else:
            rec = {
                "id": f"gen-{fp}",
                "domain": raw.get("domain", "fact"),
                "question": q,
                "correct": c,
                "wrong": w,
                "notes": str(raw.get("notes", "")),
            }
            accepted.append(rec)
            seen.add(fp)
    return accepted, rejected


def cmd_generate(args: argparse.Namespace) -> int:
    brief = DOMAIN_BRIEF[args.domain]
    prompt = TEMPLATE.format(n=args.n, brief=brief)
    got: list[dict] = []
    for attempt in range(args.rounds):
        text = call_model(prompt, args.provider, args.model)
        batch = extract_json(text)
        for b in batch:
            b["domain"] = args.domain
        got.extend(batch)
        print(f"  round {attempt + 1}: {len(batch)} items")
        if len(got) >= args.n:
            break
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for b in got:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"wrote {len(got)} raw candidates -> {args.out}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    existing: set[str] = set()
    if args.against and os.path.exists(args.against):
        with open(args.against, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r.setdefault("question", "")
                    existing.add(fingerprint(r))
    cands: list[dict] = []
    with open(args.inp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cands.append(json.loads(line))
    accepted, rejected = audit(cands, existing)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in accepted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"accepted {len(accepted)} / {len(cands)} -> {args.out}")
    for q, why in rejected[:20]:
        print(f"  rejected: {q} ({why})")
    if len(rejected) > 20:
        print(f"  ... and {len(rejected) - 20} more")
    print("\nHuman review is still required before these enter the pool.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--domain", choices=sorted(DOMAIN_BRIEF), required=True)
    g.add_argument("--n", type=int, default=40)
    g.add_argument("--rounds", type=int, default=3)
    g.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    g.add_argument("--model", default="gpt-4o-mini")
    g.add_argument("--out", default="data/candidates.jsonl")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("audit")
    a.add_argument("--inp", default="data/candidates.jsonl")
    a.add_argument("--out", default="data/accepted.jsonl")
    a.add_argument("--against", default="data/items.jsonl")
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
