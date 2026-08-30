"""Report generation: results tables plus a 'Show Your Work' disclosure card."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Sequence

from .runner import Experiment
from .stats import AxisResult
from .types import Judgment


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fingerprint(path: str) -> tuple[str, str]:
    """Identify the item set by content, not by where it happened to sit.

    Absolute paths leak the author's directory layout and help nobody verify
    anything. A content hash does: two people comparing results can tell whether
    they used the same item set.
    """
    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        return os.path.basename(path), "unreadable"
    display = path
    try:
        display = os.path.relpath(path, os.getcwd())
    except ValueError:
        pass
    return display, digest


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if lo != lo or hi != hi:  # NaN
        return "n/a"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def results_table(results: Sequence[AxisResult]) -> str:
    head = (
        "| Axis | Hyp | pairs | P(→wrong) | P(→right) | net bias | Cohen h | 95% CI | p | p_adj "
        "| noise floor | verdict |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"
    )
    rows = [head]
    for r in results:
        rows.append(
            "| `{axis}` | {hyp} | {n} | {pw} | {pr} | {nb:+.3f} | {h:+.3f} | {ci} | {p:.4f} "
            "| {pa:.4f} | {nf} | {verdict} |".format(
                axis=r.axis,
                hyp=r.hypothesis.split(":")[0],
                n=r.n_pairs,
                pw=_pct(r.p_flip_wrong),
                pr=_pct(r.p_flip_right),
                nb=r.net_bias,
                h=r.cohens_h,
                ci=_fmt_ci(r.h_ci),
                p=r.p_mcnemar,
                pa=r.p_adjusted,
                nf=_pct(r.noise_floor) if r.noise_floor else "—",
                verdict=r.verdict or "—",
            )
        )
    return "\n".join(rows)


def quality_table(judgments: Sequence[Judgment]) -> str:
    if not judgments:
        return "_no judgments recorded_"
    n = len(judgments)
    fails = sum(1 for j in judgments if j.verdict.value == "PARSE_FAIL")
    ties = sum(1 for j in judgments if j.verdict.value == "TIE")
    errs = sum(1 for j in judgments if j.error)
    lat = sorted(j.latency_ms for j in judgments)
    med = lat[len(lat) // 2] if lat else 0.0
    acc = [j.is_correct for j in judgments if j.is_scorable]
    base_acc = sum(acc) / len(acc) if acc else 0.0
    return (
        "| metric | value |\n|---|---|\n"
        f"| judgments recorded | {n} |\n"
        f"| parse failures | {fails} ({_pct(fails / n)}) |\n"
        f"| ties | {ties} ({_pct(ties / n)}) |\n"
        f"| transport errors | {errs} |\n"
        f"| median latency | {med:.0f} ms |\n"
        f"| overall accuracy | {_pct(base_acc)} |"
    )


def harness_card(exp: Experiment, judgments: Sequence[Judgment]) -> str:
    cfg = exp.config
    versions = sorted({j.model_version for j in judgments if j.model_version})
    item_display, item_hash = _fingerprint(cfg.items_path)
    return (
        "| field | value |\n|---|---|\n"
        f"| run id | `{exp.name}` |\n"
        f"| generated (UTC) | {datetime.now(timezone.utc).isoformat(timespec='seconds')} |\n"
        f"| items | {len(exp.items)} |\n"
        f"| axes | {', '.join(cfg.axes)} |\n"
        f"| replicates per condition | {cfg.reps} |\n"
        f"| temperature | {cfg.temperature} |\n"
        f"| prompt templates | {', '.join(cfg.templates)} |\n"
        f"| noise floor | {'yes (2 unperturbed repeats)' if cfg.include_noise_floor else 'no'} |\n"
        f"| seed | {cfg.seed} |\n"
        f"| judge ids | {', '.join(j.id for j in exp.judges)} |\n"
        f"| model version strings | {', '.join(versions) if versions else 'not reported'} |\n"
        f"| item set | `{item_display}` ({len(exp.items)} items) |\n"
        f"| item set sha256[:12] | `{item_hash}` |"
    )


def write_report(
    exp: Experiment,
    analyses: dict[str, list[AxisResult]],
    path: str,
    judgments: Sequence[Judgment] | None = None,
) -> str:
    judgments = list(judgments) if judgments is not None else exp.load_judgments()
    lines: list[str] = [
        f"# judge-redteam results — `{exp.name}`",
        "",
        "Direction decomposition: **net bias = P(correct→wrong) − P(wrong→correct)**. "
        "Zero means the perturbation only adds noise; positive means it steers the "
        "judge away from ground truth.",
        "",
        "## Harness disclosure",
        "",
        harness_card(exp, judgments),
        "",
        "## Response quality",
        "",
        quality_table(judgments),
        "",
    ]
    for judge_id, results in analyses.items():
        lines += [f"## Judge `{judge_id}`", "", results_table(results), ""]
    lines += [
        "## Notes",
        "",
        "- p-values are two-sided exact McNemar on discordant pairs.",
        "- p_adj is Holm–Bonferroni across the confirmatory hypotheses in this run.",
        "- An axis flagged *below noise floor* flipped fewer verdicts than the judge "
        "flips on its own between two unperturbed repeats; it is not reported as a finding.",
        "- Null results carry the same weight as positive ones and are retained.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


__all__ = ["results_table", "quality_table", "harness_card", "write_report"]
