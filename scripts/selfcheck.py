"""Calibrate the measuring instrument.

A bias detector is only credible if it has been shown to (a) recover a bias of
known magnitude and (b) stay quiet when there is no bias. This script runs both
directions against SimulatedJudge and prints a pass/fail table.

Run:  python scripts/selfcheck.py
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jrt.judges import BiasProfile, SimulatedJudge  # noqa: E402
from jrt.report import results_table  # noqa: E402
from jrt.runner import Experiment, RunConfig  # noqa: E402
from jrt.stats import cohens_h, holm_bonferroni, mcnemar_exact  # noqa: E402

ITEMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "items.jsonl"
)

# A judge at competence 0.65 with noise 0.35 sits at roughly 88% unperturbed
# accuracy, inside the [0.55, 0.90] band the preregistration requires — outside
# it, ceiling or floor effects make every axis look clean.
#
# Injected biases are set near the detection threshold on purpose. A bias so
# large that it flips every verdict proves nothing about sensitivity; the
# interesting question is whether the pipeline catches a bias comparable in
# magnitude to the judge's own noise.

UNBIASED = BiasProfile(competence=0.65, competence_spread=0.22, noise=0.35)

BIASED = BiasProfile(
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

AXES_ALL = (
    "position",
    "length",
    "authority",
    "format",
    "verbose_cot",
    "length_matched_control",
    "abstention",
    "self_preference",
)


def check_primitives() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    # McNemar: 30 vs 10 discordant pairs is strongly asymmetric.
    p = mcnemar_exact(30, 10)
    out.append(("mcnemar detects 30/10 asymmetry", p < 0.01, f"p={p:.5f}"))

    # Symmetric discordance must not be significant.
    p = mcnemar_exact(20, 20)
    out.append(("mcnemar silent on 20/20 symmetry", p > 0.5, f"p={p:.4f}"))

    # No discordance at all.
    out.append(("mcnemar returns 1.0 when 0/0", mcnemar_exact(0, 0) == 1.0, "p=1.0"))

    # Cohen's h sign and symmetry.
    h = cohens_h(0.6, 0.2)
    out.append(("cohens_h positive when p1>p2", h > 0, f"h={h:+.4f}"))
    out.append(("cohens_h antisymmetric", abs(cohens_h(0.2, 0.6) + h) < 1e-12, "ok"))
    out.append(("cohens_h zero at equality", abs(cohens_h(0.4, 0.4)) < 1e-12, "ok"))

    # Holm: three p-values, smallest should survive, largest should not.
    rejected, adjusted = holm_bonferroni([0.001, 0.04, 0.03], alpha=0.05)
    out.append(("holm rejects the smallest p", rejected[0] is True, f"adj={adjusted[0]:.4f}"))
    out.append(("holm adjusts monotonically", adjusted[0] <= adjusted[2] <= adjusted[1], "ok"))
    out.append(("holm caps adjustment at 1.0", all(a <= 1.0 for a in adjusted), "ok"))

    rejected, _ = holm_bonferroni([0.04, 0.04], alpha=0.05)
    out.append(("holm tighter than raw alpha on ties", rejected == [False, False], f"{rejected}"))
    return out


def run_condition(name: str, profile: BiasProfile, seed: int) -> dict:
    judge = SimulatedJudge(profile=profile, seed=seed, name=f"sim-{name}")
    cfg = RunConfig(
        items_path=ITEMS,
        axes=AXES_ALL,
        reps=5,
        temperature=0.7,
        seed=seed,
        output_dir=os.path.join(os.path.dirname(ITEMS), "..", "results"),
        run_name=f"selfcheck-{name}",
        resume=False,
    )
    exp = Experiment(cfg, [judge])
    exp.run(verbose=False)
    judgments = exp.load_judgments()
    analyses = exp.analyse(judgments)
    return {
        "name": name,
        "judge": judge,
        "results": analyses[judge.id],
        "judgments": judgments,
    }


def evaluate_negative(cond: dict) -> list[tuple[str, bool, str]]:
    """Family-wise criterion: under zero injected bias, nothing should survive
    Holm correction. Per-axis, an uncorrected p near 0.05 is expected to show up
    occasionally — demanding zero would be demanding a Type I error rate of zero,
    which no valid test delivers. What must hold is that the correction catches
    it, and that across seeds the rejection rate stays near alpha."""
    out = []
    for r in cond["results"]:
        out.append(
            (
                f"unbiased judge / {r.axis}: no false positive",
                not r.rejected,
                f"net_bias={r.net_bias:+.3f} p={r.p_mcnemar:.3f} p_adj={r.p_adjusted:.3f}",
            )
        )
    return out


def check_false_positive_rate(conds: Sequence[dict]) -> list[tuple[str, bool, str]]:
    """Aggregate across seeds: the family-wise rejection rate under the null."""
    total = rejected = 0
    per = []
    for c in conds:
        n = len(c["results"])
        k = sum(1 for r in c["results"] if r.rejected)
        total += n
        rejected += k
        per.append(f"{k}/{n}")
    rate = rejected / total if total else 0.0
    return [
        (
            "family-wise false positive rate under zero bias",
            rate <= 0.05,
            f"{'/'.join(per)} -> {rejected}/{total} = {rate:.1%} (target <= 5%)",
        )
    ]


def evaluate_positive(cond: dict) -> list[tuple[str, bool, str]]:
    out = []
    for r in cond["results"]:
        ok = r.rejected and r.net_bias > 0
        out.append(
            (
                f"biased judge / {r.axis}: bias recovered",
                ok,
                f"net_bias={r.net_bias:+.3f} h={r.cohens_h:+.3f} p_adj={r.p_adjusted:.4f} -> {r.verdict}",
            )
        )
    return out


def evaluate_ablation(cond: dict) -> list[tuple[str, bool, str]]:
    """H5 rests on the claim that it is the reasoning *structure*, not the extra
    length, that moves the judge. The length-matched control pads the wrong
    answer to the same size with text that says nothing. If the verbose chain
    does not beat that control, H5 has not been isolated."""
    by = {r.axis: r for r in cond["results"]}
    cot, ctrl = by.get("verbose_cot"), by.get("length_matched_control")
    if not cot or not ctrl:
        return [("H5 ablation present", False, "control axis missing from run")]
    delta = cot.net_bias - ctrl.net_bias
    return [
        (
            "H5 ablation: reasoning structure beats length alone",
            delta > 0,
            f"cot {cot.net_bias:+.3f} vs length-matched {ctrl.net_bias:+.3f} (delta {delta:+.3f})",
        )
    ]


def main() -> int:
    checks = check_primitives()
    print("=" * 78)
    print("STATISTICAL PRIMITIVES")
    print("=" * 78)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<48} {detail}")

    print()
    print("=" * 78)
    print("CONDITION A — judge with NO injected bias (false-positive control)")
    print("=" * 78)
    neg = run_condition("unbiased", UNBIASED, seed=101)
    print(results_table(neg["results"]))
    neg_checks = evaluate_negative(neg)
    print()
    for label, ok, detail in neg_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<48} {detail}")

    # Two extra seeds so the false-positive rate is estimated over 24 tests
    # rather than 8. A single run cannot distinguish a calibrated instrument
    # from a lucky one.
    extra = [run_condition(f"unbiased-s{s}", UNBIASED, seed=s) for s in (202, 303)]
    fpr_checks = check_false_positive_rate([neg] + extra)
    checks += neg_checks + fpr_checks
    print()
    for label, ok, detail in fpr_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<48} {detail}")

    print()
    print("=" * 78)
    print("CONDITION B — judge WITH injected bias (sensitivity control)")
    print("=" * 78)
    pos = run_condition("biased", BIASED, seed=202)
    print(results_table(pos["results"]))
    pos_checks = evaluate_positive(pos) + evaluate_ablation(pos)
    checks += pos_checks
    print()
    for label, ok, detail in pos_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<48} {detail}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print()
    print("=" * 78)
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed")
    print("=" * 78)
    if n_fail:
        print("Instrument is NOT calibrated. Do not trust results from real judges.")
        return 1
    print("Instrument calibrated: recovers known bias, silent when bias is absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
