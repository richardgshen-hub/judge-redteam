"""Optional, source-backed static figures for Markdown reports.

Matplotlib is imported only when rendering. The experiment runner stays
standard-library only. Public exports use an allowlist, never a raw manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

LABELS = {
    "position": "H1  Position",
    "length": "H2  Length",
    "authority": "H3  Authority",
    "format": "H4  Formatting",
    "verbose_cot": "H5  Verbose reasoning",
    "abstention": "H6  Abstention",
    "self_preference": "H7  Attribution",
    "length_matched_control": "H5  Length control",
}
INK = "#16283D"
MUTED = "#526276"
BLUE = "#245FC4"
ORANGE = "#BD4D26"
TEAL = "#117A70"
GRID = "#E4E9F0"
FIELDS = (
    "axis", "hypothesis", "n_pairs", "n_items", "b", "c", "net_bias",
    "p_flip_wrong", "p_flip_right", "cohens_h", "h_ci", "p_mcnemar",
    "p_adjusted", "noise_floor", "above_noise", "confirmatory", "verdict",
)
START = "<!-- jrt:figures:start -->"
END = "<!-- jrt:figures:end -->"


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def public_source(summary: dict, judge_id: str | None = None) -> dict:
    """Select one judge explicitly and export only fields used by the figures."""
    results = summary.get("results", {})
    if not results:
        raise ValueError("Summary contains no results; run an experiment first.")
    if judge_id is None:
        if len(results) != 1:
            raise ValueError("Multiple judges found; select one with --judge.")
        judge_id = next(iter(results))
    if judge_id not in results or not results[judge_id]:
        raise ValueError(f"No result rows for judge {judge_id!r}.")
    manifest = summary.get("manifest", {})
    config = summary.get("config", manifest.get("config", {}))
    judges = manifest.get("judges", summary.get("judges", []))
    identity = next((j for j in judges if j.get("id") == judge_id), {})
    kind = identity.get("type", "unknown")
    rows = [{k: row[k] for k in FIELDS if k in row} for row in results[judge_id]]
    source = {
        "schema": "jrt-figure-source-v1",
        "run_id": summary.get("run_id", summary.get("run", "unknown")),
        "generated_utc": summary.get("generated_utc"),
        "n_items": summary.get("n_items", manifest.get("items_count")),
        "config": {k: config[k] for k in ("axes", "reps", "temperature", "templates", "seed", "include_noise_floor") if k in config},
        "manifest": {
            "items_sha256": manifest.get("items_sha256"),
            "code_commit": manifest.get("code_commit", "unknown"),
            "judges": [{"id": judge_id, "type": kind}],
        },
        "results": {judge_id: rows},
    }
    return _clean(source)


def pool_records(items_path: Path) -> list[dict]:
    records = []
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        a, b = (len(re.sub(r"\s+", " ", item[k]).strip()) for k in ("correct", "wrong"))
        records.append({"item_id": item["id"], "domain": item["domain"], "signed_length_gap": (a - b) / max(a, b, 1)})
    if not records:
        raise ValueError("Item pool is empty.")
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row_export(row: dict) -> dict:
    result = {k: row.get(k) for k in FIELDS if k != "h_ci"}
    lo, hi = row.get("h_ci") or [None, None]
    result.update(h_ci_low=lo, h_ci_high=hi)
    return result


def _label(row: dict) -> str:
    return LABELS.get(row["axis"], row["axis"].replace("_", " "))


def _status(source: dict) -> str:
    kind = source["manifest"]["judges"][0]["type"]
    return "SIMULATED DATA" if kind == "SimulatedJudge" else "RUN DATA / CHECK PROVENANCE"


def _frame(plt, source, number, title, subtitle, mobile=False, height=7.0, status=None):
    fig = plt.figure(figsize=(6.2 if mobile else 11.5, 9.2 if mobile else height), facecolor="white")
    fig.text(.045, .952, f"JUDGE-REDTEAM   /   {number}   /   {status or _status(source)}", color=BLUE, fontsize=10, weight="bold")
    fig.text(.045, .886, title, color=INK, fontsize=21 if mobile else 25, weight="bold", va="top")
    if mobile:
        subtitle = "\n".join(textwrap.fill(line, width=59) for line in subtitle.splitlines())
    fig.text(.045, .762 if mobile else .802, subtitle, color=MUTED, fontsize=11, va="top", linespacing=1.5)
    run = source.get("run_id", "unknown")
    seed = source.get("config", {}).get("seed", "unknown")
    fig.text(.045, .028, f"Run: {run}   |   seed: {seed}   |   source.json + axis_results.csv", fontsize=8.5, color=MUTED)
    return fig


def _axes(fig, mobile, n):
    ax = fig.add_axes([.38 if mobile else .25, .27, .50 if mobile else .59, .35 if mobile else .40])
    ax.set_ylim(n - .5, -.7)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.tick_params(axis="both", length=0, labelsize=11, labelcolor=MUTED, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _save(fig, output: Path, stem: str, mobile=False):
    if mobile:
        for text in fig.texts:
            if text.get_position()[1] <= .15 and text.get_position()[1] > .04:
                text.set_text("\n".join(textwrap.fill(line, width=62) for line in text.get_text().splitlines()))
    name = stem + ("_mobile" if mobile else "")
    fig.savefig(output / f"{name}.svg", metadata={"Date": None, "Creator": "judge-redteam / matplotlib"})
    if not mobile:
        fig.savefig(output / f"{name}.png", dpi=180, metadata={"Software": "judge-redteam / matplotlib"})


def _effects(plt, source, output, mobile):
    rows = next(iter(source["results"].values()))
    title = "Directional effects,\nwith uncertainty" if mobile else "Directional effects, with uncertainty"
    fig = _frame(plt, source, "01", title, "Cohen's h; 95% item-cluster bootstrap intervals.\nPositive = more correct-to-wrong than wrong-to-correct flips.", mobile)
    ax = _axes(fig, mobile, len(rows))
    values = [0.0]
    for row in rows:
        values.extend(v for v in [row.get("cohens_h"), *(row.get("h_ci") or [])] if finite(v))
    low, high = min(values), max(values)
    span = max(high-low, .4)
    ax.set_xlim(low-.08*span, high+.18*span)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    for i, row in enumerate(rows):
        h = row.get("cohens_h")
        control = not row.get("confirmatory", True)
        color = MUTED if control else BLUE
        ci = row.get("h_ci") or []
        if len(ci) == 2 and all(finite(v) for v in ci):
            ax.plot(ci, [i, i], color=color, linewidth=2.1, solid_capstyle="round")
            ax.plot(ci, [i, i], "|", color=color, markersize=9)
        if finite(h):
            ax.plot(h, i, "D" if control else "o", color=color, markersize=7)
        label = f"{h:+.2f}" if finite(h) else "n/a"
        ax.text(1.025, i, label, transform=ax.get_yaxis_transform(), va="center", fontsize=11, color=color, weight="bold")
        if control:
            ax.axhspan(i-.45, i+.45, color=GRID, alpha=.35, zorder=-1)
    ax.set_yticks(range(len(rows)), [_label(r).replace("Verbose reasoning", "Verbose\nreasoning").replace("Length control", "Length\ncontrol") if mobile else _label(r) for r in rows])
    ax.set_xlabel("Cohen's h (directional effect)", color=MUTED, fontsize=11, labelpad=13)
    fig.text(.045, .105, "Intervals are unadjusted; they are not the Holm decision rule.\nDiamond = exploratory H5 length control, outside the seven-test family.", fontsize=10, color=MUTED, linespacing=1.5)
    _save(fig, output, "effect_sizes", mobile)
    plt.close(fig)


def _flips(plt, source, output, mobile):
    rows = next(iter(source["results"].values()))
    fig = _frame(plt, source, "02", "Separate harm\nfrom recovery" if mobile else "Separate harm from recovery", "Two flip directions, with an unperturbed reference.\nRates are shares of paired scorable trials, not conditional risks.", mobile)
    ax = _axes(fig, mobile, len(rows))
    series = [("p_flip_wrong", ORANGE, -.14), ("p_flip_right", TEAL, .14)]
    for key, color, offset in series:
        for i, row in enumerate(rows):
            if finite(row.get(key)):
                ax.barh(i+offset, row[key]*100, height=.23, color=color)
                ax.text(row[key]*100+.5, i+offset, f"{row[key]:.1%}", va="center", color=color, fontsize=9)
    show_noise = source.get("config", {}).get("include_noise_floor", False)
    if show_noise:
        for i, row in enumerate(rows):
            if finite(row.get("noise_floor")):
                ax.plot(row["noise_floor"]*100, i, "D", markersize=6, markerfacecolor="white", markeredgecolor=INK, zorder=5)
    max_rate = max([r.get(k) or 0 for r in rows for k in ("p_flip_wrong", "p_flip_right", "noise_floor")])
    ax.set_xlim(0, max(20, max_rate*100+9))
    ax.set_yticks(range(len(rows)), [_label(r).replace("Verbose reasoning", "Verbose\nreasoning").replace("Length control", "Length\ncontrol") if mobile else _label(r) for r in rows])
    ax.set_xlabel("Share of paired scorable trials (%)", color=MUTED, fontsize=11, labelpad=13)
    fig.text(.045, .650 if mobile else .713, "Correct → wrong", color=ORANGE, fontsize=11, weight="bold")
    fig.text(.46 if mobile else .30, .650 if mobile else .713, "Wrong → correct", color=TEAL, fontsize=11, weight="bold")
    fig.text(.045, .105, ("◇ Unperturbed wrong-flip reference; rates are descriptive.\nThe noise gate uses a paired item-level bootstrap, not these bar gaps." if show_noise else "No noise-control data in this run.\nH5 length control is exploratory; retain null and negative effects."), fontsize=10, color=MUTED, linespacing=1.5)
    _save(fig, output, "directional_flips", mobile)
    plt.close(fig)


def _pool(plt, source, pool, output, mobile):
    domains = sorted(Counter(r["domain"] for r in pool))
    fig = _frame(plt, source, "03", "Inspect answer-length\nbalance" if mobile else "Inspect answer-length balance", f"{len(pool)} constructed items across {len(domains)} domains; each dot is one item.\nSigned gap = (correct length − wrong length) / longer length.", mobile, status="ITEM-POOL AUDIT")
    ax = _axes(fig, mobile, len(domains))
    extent = max(.185, max(abs(r["signed_length_gap"]) for r in pool)+.035)
    ax.set_xlim(-extent, extent)
    ax.axvspan(-.15, .15, color=BLUE, alpha=.04)
    for threshold in (-.15, .15):
        ax.axvline(threshold, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    ax.axvline(0, color=INK, linewidth=.7)
    labels = []
    for i, domain in enumerate(domains):
        group = sorted([r for r in pool if r["domain"] == domain], key=lambda r: hashlib.sha256(r["item_id"].encode()).hexdigest())
        labels.append(f"{domain.capitalize()}\nn = {len(group)}")
        for j, row in enumerate(group):
            # Deterministic vertical displacement is only to separate marks.
            offset = (j / max(len(group)-1, 1) - .5) * .65
            ax.scatter(row["signed_length_gap"], i+offset, color=BLUE, s=22, alpha=.65, edgecolors="none")
    ax.set_yticks(range(len(domains)), labels)
    ax.set_xticks([-.15, 0, .15], ["−15%", "0", "+15%"])
    ax.set_xlabel("Wrong longer  ←  signed gap  →  correct longer", color=MUTED, fontsize=10, labelpad=13)
    mean = sum(r["signed_length_gap"] for r in pool)/len(pool)
    fails = sum(abs(r["signed_length_gap"]) > .15 for r in pool)
    fig.text(.045, .105, f"Mean signed gap: {mean:+.2%}  ·  Outside ±15%: {fails}/{len(pool)}\nWhitespace-normalized characters. Length balance is not semantic validation.", fontsize=10, color=MUTED, linespacing=1.5)
    _save(fig, output, "item_pool", mobile)
    plt.close(fig)


def _protocol(plt, source, pool, output, mobile):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    n_axes = len(next(iter(source["results"].values())))
    fig = _frame(plt, source, "00", "Audit the judge.\nTrace every verdict." if mobile else "Audit the judge. Trace every verdict.", "A paired experiment, from controlled inputs to inspectable evidence.\nSchematic of the current harness; the example run is simulated.", mobile, status="EXPERIMENT DESIGN")
    ax = fig.add_axes([.045, .13, .91, .50 if mobile else .57])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("01", "Construct the item pool", f"{len(pool)} items / {len(set(r['domain'] for r in pool))} domains · length-balance gate"),
        ("02", "Build paired conditions", f"Base + intervention{' + noise A/B' if source.get('config', {}).get('include_noise_floor') else ''} · {n_axes} axes in this run"),
        ("03", "Collect judge verdicts", "Repeated calls · raw responses · manifest + data hash"),
        ("04", "Analyse at the item level", "Directional flips · cluster intervals · Holm + noise gate"),
        ("05", "Publish inspectable evidence", "Figures + tables · exclusions · null results · provenance"),
    ]
    for i, (num, title, body) in enumerate(steps):
        y = .84-i*.19
        ax.add_patch(FancyBboxPatch((.10, y), .89, .15, boxstyle="round,pad=0.005,rounding_size=0.018", facecolor="#F3F6FB", edgecolor="none"))
        ax.text(.0, y+.088, num, color=BLUE, fontsize=17, weight="bold", va="center")
        ax.text(.13, y+.10, title, color=INK, fontsize=12 if mobile else 14, weight="bold", va="center")
        if mobile:
            body = body.replace(" · ", "\n", 1)
        ax.text(.13, y+.045, body, color=MUTED, fontsize=9 if mobile else 11, va="center")
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((.052, y-.007), (.052, y-.045), arrowstyle="-|>", mutation_scale=11, color=MUTED, linewidth=.8))
    _save(fig, output, "protocol", mobile)
    plt.close(fig)


def embed_figures(report_path: Path, output: Path) -> None:
    """Insert or replace only the generated figure section; preserve report text."""
    import os

    prefix = Path(os.path.relpath(output, report_path.parent)).as_posix()
    sections = [START, "## Visual analysis", "", "Figures use the same summary as the tables. Read each figure's evidence label.", ""]
    for stem, alt in [
        ("effect_sizes", "Directional effect sizes with unadjusted 95% item-cluster bootstrap intervals; H5 length control is exploratory."),
        ("directional_flips", "Correct-to-wrong and wrong-to-correct flip rates with descriptive unperturbed noise references."),
        ("item_pool", "Per-item answer-length gaps by domain; semantic validation remains pending."),
    ]:
        sections += [f'<picture>\n  <source media="(max-width: 600px)" srcset="{prefix}/{stem}_mobile.svg">\n  <img src="{prefix}/{stem}.svg" alt="{alt}">\n</picture>', ""]
    sections += [f"[Figure data]({prefix}/source.json) · [Axis values (CSV)]({prefix}/axis_results.csv) · [Item values (CSV)]({prefix}/item_pool.csv)", "", "Intervals are unadjusted. Aggregate flip/noise rates are descriptive; the noise gate uses paired item-level differences. The exploratory H5 control is outside the seven-hypothesis Holm family.", END]
    section = "\n".join(sections)
    text = report_path.read_text(encoding="utf-8")
    if START in text and END in text:
        before, remainder = text.split(START, 1)
        _, after = remainder.split(END, 1)
        text = before + section + after
    elif "## Harness disclosure" in text:
        text = text.replace("## Harness disclosure", section+"\n\n## Harness disclosure", 1)
    else:
        text += "\n"+section+"\n"
    report_path.write_text(text, encoding="utf-8")


def render_figures(summary_path: Path, items_path: Path, output: Path, judge_id: str | None = None, report_path: Path | None = None) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Figure rendering requires: python -m pip install -e ".[viz]"') from exc

    source = public_source(json.loads(summary_path.read_text(encoding="utf-8")), judge_id)
    digest = hashlib.sha256(items_path.read_bytes()).hexdigest()
    expected = source["manifest"].get("items_sha256")
    if not expected or digest != expected:
        raise ValueError("Item-pool hash does not match the run; refusing mismatched figures.")
    pool = pool_records(items_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source.json").write_text(json.dumps(source, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    _write_csv(output / "axis_results.csv", [_row_export(r) for r in next(iter(source["results"].values()))])
    _write_csv(output / "item_pool.csv", pool)
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "judge-redteam-v0.2", "axes.unicode_minus": True}):
        for mobile in (False, True):
            _effects(plt, source, output, mobile)
            _flips(plt, source, output, mobile)
            _pool(plt, source, pool, output, mobile)
            _protocol(plt, source, pool, output, mobile)
    if report_path is not None:
        embed_figures(report_path, output)
    return sorted(output.glob("*.svg"))
