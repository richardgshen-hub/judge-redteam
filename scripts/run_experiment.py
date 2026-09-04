"""Run the preregistered experiment against judges.

Pilot (cheap, a few minutes, local or low-cost):

  python scripts/run_experiment.py --pilot --judge ollama --model qwen2.5:7b

Formal run (preregistered configuration, costs real calls on paid backends):

  python scripts/run_experiment.py --judge ollama --model qwen2.5:7b
  python scripts/run_experiment.py --judge openai --model gpt-4o-mini --reps 5 --yes

Always prints the estimated call count before doing anything. Paid backends
(openai, anthropic) additionally require an explicit `--yes` (or an interactive
confirmation), and `--dry-run` shows the plan without issuing a single call.
The default backend is local Ollama — no paid API is ever contacted by default.

Every trial is persisted (and fsynced) as it completes; re-running the same
command resumes where it stopped. Calls are sequential with an optional rate
limit (`--rpm`); parallel execution is deliberately not offered, because the
crash-resume guarantee depends on the write ordering.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jrt import Experiment, RunConfig  # noqa: E402
from jrt.axes import AXES, CONFIRMATORY_AXES  # noqa: E402
from jrt.judges import (  # noqa: E402
    AnthropicJudge,
    OllamaJudge,
    OpenAICompatJudge,
    SimulatedJudge,
)
from jrt.prompts import TEMPLATES  # noqa: E402
from jrt.report import write_report  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAID_BACKENDS = {"openai", "anthropic"}


def build_judges(args: argparse.Namespace) -> list:
    if args.judge == "ollama":
        return [OllamaJudge(model=args.model, host=args.host)]
    if args.judge == "openai":
        return [OpenAICompatJudge(model=args.model, base_url=args.base_url)]
    if args.judge == "anthropic":
        return [AnthropicJudge(model=args.model)]
    if args.judge == "simulated":
        return [SimulatedJudge(name="sim")]
    raise SystemExit(f"unknown judge backend {args.judge!r}")


def _confirm_paid(args: argparse.Namespace, estimated_calls: int) -> None:
    """Paid backends never start without an explicit, informed yes."""
    if args.judge not in PAID_BACKENDS:
        return
    if args.yes:
        return
    print(
        f"\nThis uses the PAID backend '{args.judge}'. Estimated judge calls: "
        f"{estimated_calls:,}. Re-running the same command later resumes for free; "
        "interrupted work is never re-billed."
    )
    answer = input("Proceed with the paid run? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Aborted before any API call was made. Use --dry-run to inspect the plan.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="ollama", choices=["ollama", "openai", "anthropic", "simulated"])
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--items", default=os.path.join(ROOT, "data", "items.jsonl"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--axes", nargs="*", default=list(CONFIRMATORY_AXES))
    ap.add_argument("--templates", nargs="*", default=["v1_standard"],
                    help=f"any of {sorted(TEMPLATES)}; analysed separately per template")
    ap.add_argument("--pilot", action="store_true",
                    help="cheap pilot: 2 axes, 1 template, 1 rep. For checking the plumbing, not for findings.")
    ap.add_argument("--rpm", type=float, default=0.0,
                    help="max requests per minute (rate limit). 0 = unlimited.")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and estimated call count, then exit without any judge call")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive confirmation for paid backends")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    bad_axes = [a for a in args.axes if a not in AXES]
    if bad_axes:
        raise SystemExit(f"unknown axes {bad_axes}; available: {sorted(AXES)}")
    bad_tpl = [t for t in args.templates if t not in TEMPLATES]
    if bad_tpl:
        raise SystemExit(f"unknown templates {bad_tpl}; available: {sorted(TEMPLATES)}")

    if args.pilot:
        args.axes = list(args.axes)[:2]
        args.templates = args.templates[:1]
        args.reps = 1

    cfg = RunConfig(
        items_path=args.items,
        axes=tuple(args.axes),
        reps=args.reps,
        temperature=args.temperature,
        templates=tuple(args.templates),
        seed=args.seed,
        output_dir=args.out,
        run_name=args.name,
        resume=not args.no_resume,
    )
    judges = build_judges(args)
    exp = Experiment(cfg, judges)

    estimated = exp.expected_calls()
    print(f"[plan] judge={args.judge} model={args.model}")
    print(f"[plan] items={len(exp.items)} axes={len(cfg.axes)} templates={list(cfg.templates)} "
          f"reps={cfg.reps} conditions={4 if cfg.include_noise_floor else 2}")
    print(f"[plan] estimated judge calls: {estimated:,}")
    if args.rpm > 0:
        print(f"[plan] rate limit: {args.rpm:g} req/min")

    if args.dry_run:
        print("[dry-run] no calls were made.")
        return 0

    _confirm_paid(args, estimated)

    min_interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    exp.run(min_interval=min_interval)

    judgments = exp.load_judgments()
    analyses = exp.analyse(judgments)
    summary = exp.write_summary(analyses)
    report = os.path.join(cfg.output_dir, f"{exp.name}_report.md")
    write_report(exp, analyses, report, judgments)

    print()
    for judge_id, results in analyses.items():
        print(f"--- {judge_id} ---")
        for r in results:
            flag = "BIAS" if r.rejected and r.above_noise is not False else "  . "
            print(
                f"  [{flag}] {r.axis:<24} net={r.net_bias:+.3f} h={r.cohens_h:+.3f} "
                f"p_adj={r.p_adjusted:.4f}  {r.verdict}"
            )
    print()
    print(f"summary: {summary}")
    print(f"report:  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
