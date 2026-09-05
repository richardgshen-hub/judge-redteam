"""Generate a full worked example report with no API key and no network.

The judge here is simulated with known biases injected, so the output is not a
finding about any real model — it is a demonstration of what a completed run
looks like, and a check that the reporting path works end to end.

Run:  python scripts/demo.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jrt import Experiment, RunConfig  # noqa: E402
from jrt.axes import CONFIRMATORY_AXES  # noqa: E402
from jrt.judges import BiasProfile, SimulatedJudge  # noqa: E402
from jrt.report import write_report  # noqa: E402
from jrt.stats import AxisResult  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known injected biases exercise the report. Their apparent strengths are
# properties of this simulation, not estimates for any real model.
PROFILE = BiasProfile(
    competence=0.65,
    competence_spread=0.22,
    noise=0.35,
    position=0.30,
    length=0.30,
    authority=0.30,
    format=0.30,
    verbose_cot=0.15,
    abstention=0.45,
    self_preference=0.42,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="Also render SVG/PNG report figures (requires .[viz])")
    args = parser.parse_args()
    judge = SimulatedJudge(profile=PROFILE, seed=20260830, name="demo-simulated-judge")
    cfg = RunConfig(
        items_path=os.path.join(ROOT, "data", "items.jsonl"),
        axes=CONFIRMATORY_AXES + ("length_matched_control",),
        reps=5,
        temperature=0.7,
        output_dir=os.path.join(ROOT, "results"),
        run_name="demo",
        resume=False,
    )
    exp = Experiment(cfg, [judge])
    print("running demo experiment against a simulated judge with known biases...")
    exp.run(verbose=False)
    judgments = exp.load_judgments()
    analyses: dict[str, list[AxisResult]] = exp.analyse(judgments)
    summary = exp.write_summary(analyses)
    report = write_report(exp, analyses, os.path.join(ROOT, "results", "demo_report.md"), judgments)
    if args.figures:
        from jrt.figures import render_figures

        render_figures(Path(summary), Path(cfg.items_path), Path(ROOT) / "docs/figures/demo", report_path=Path(report))

    print()
    for judge_id, results in analyses.items():
        print(f"--- {judge_id} ---")
        for r in results:
            flag = "BIAS" if r.rejected and r.above_noise is not False else "  . "
            print(
                f"  [{flag}] {r.axis:<24} net={r.net_bias:+.3f}  h={r.cohens_h:+.3f}  "
                f"p_adj={r.p_adjusted:.4f}  {r.verdict}"
            )
    print()
    print(f"report:  {report}")
    print(f"summary: {summary}")
    print()
    print("Reminder: this judge is simulated. These numbers demonstrate the")
    print("reporting path, they are not evidence about any real model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
