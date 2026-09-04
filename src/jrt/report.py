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
        "| Axis | Hyp | pairs | items | acc base | acc pert | P(→wrong) | P(→right) "
        "| flips | net bias | Cohen h | 95% CI | p | p_adj "
        "| noise ▸wrong | verdict |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"
    )
    rows = [head]
    for r in results:
        acc_base = r.extra.get("acc_base")
        acc_pert = r.extra.get("acc_pert")
        rows.append(
            "| `{axis}` | {hyp} | {n} | {ni} | {ab} | {ap} | {pw} | {pr} | {fr} "
            "| {nb:+.3f} | {h:+.3f} | {ci} | {p:.4f} "
            "| {pa:.4f} | {nf} | {verdict} |".format(
                axis=r.axis,
                hyp=r.hypothesis.split(":")[0],
                n=r.n_pairs,
                ni=r.n_items or "—",
                ab=_pct(acc_base) if acc_base is not None else "—",
                ap=_pct(acc_pert) if acc_pert is not None else "—",
                pw=_pct(r.p_flip_wrong),
                pr=_pct(r.p_flip_right),
                fr=_pct(r.flip_rate),
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
    if len(exp.config.templates) > 1:
        by_template = exp.analyse_by_template(list(judgments))
        lines += [
            "## Per-template breakdown",
            "",
            "The pooled analysis above treats prompt templates as another replicate "
            "dimension. This section analyses each template separately, because a "
            "phrasing effect is only visible when they are not pooled.",
            "",
        ]
        for judge_id, per_template in by_template.items():
            for template, results in per_template.items():
                lines += [
                    f"### `{template}` (judge `{judge_id}`)",
                    "",
                    results_table(results),
                    "",
                ]
    lines += [
        "## Notes",
        "",
        "- p-values are item-cluster permutation tests (each item's total discordance "
        "is one cluster; direction is randomized under the null). See `stats.STATS_REVIEW_NOTE`.",
        "- p_adj is Holm–Bonferroni across the confirmatory hypotheses in this run only.",
        "- **Noise floor** in the table is the *wrong-direction self-flip rate*: how often the "
        "judge flips a correct verdict to a wrong one between two identical, unperturbed "
        "repeats (noise_a vs noise_b). A perturbation is only called a finding when its own "
        "P(correct→wrong) clearly exceeds this baseline.",
        "- The *total* self-disagreement rate (any flip, either direction) is larger and is "
        "recorded separately in the JSON summary as `noise_self_disagreement`; it is not used "
        "for the above-noise decision.",
        "- **Exclusions are not silent.** PARSE_FAIL verdicts (which include refusals with no "
        "parseable verdict), TIE verdicts, and transport-errored calls are excluded from the "
        "paired analysis and counted in *Response quality* above; every raw response is "
        "preserved in the run's `*_raw.jsonl`. `pairs` counts paired scorable trials; "
        "`items` is the number of distinct items (the effective sample size for inference).",
        "- An axis flagged *below noise floor* flipped fewer verdicts toward wrong than the "
        "judge does on its own; it is not reported as a finding even if p < alpha.",
        "- Null results carry the same weight as positive ones and are retained.",
        "- **Reminder: any run against a simulated judge demonstrates the reporting "
        "pipeline only and is not evidence about any real model.**",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


__all__ = ["results_table", "quality_table", "harness_card", "write_report"]
