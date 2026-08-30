"""Run the preregistered experiment against real judges.

  python scripts/run_experiment.py --judge ollama --model qwen2.5:7b
  python scripts/run_experiment.py --judge openai --model gpt-4o-mini --reps 5
  python scripts/run_experiment.py --judge anthropic --model claude-opus-4-6 --axes verbose_cot abstention

Costs money. Every trial is persisted as it completes, and re-running the same
command resumes where it stopped instead of starting over.
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
from jrt.report import write_report  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    ap.add_argument("--templates", nargs="*", default=["v1_standard"])
    ap.add_argument("--name", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    bad = [a for a in args.axes if a not in AXES]
    if bad:
        raise SystemExit(f"unknown axes {bad}; available: {sorted(AXES)}")

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
    exp = Experiment(cfg, build_judges(args))
    exp.run()
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
