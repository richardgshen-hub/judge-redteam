"""Experiment orchestration.

Replicates the preregistered protocol: paired base/perturbed conditions, R
replicates at temperature > 0, a noise floor from two unperturbed repeats, full
raw-response persistence, and crash-safe resume.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .axes import HYPOTHESES, get_axis
from .judges.base import Judge
from .prompts import parse_verdict
from .stats import apply_correction, summarise_axis, AxisResult
from .types import (
    Condition,
    Item,
    Judgment,
    PairedOutcome,
    Presentation,
    Trial,
    Verdict,
    dump_jsonl,
    load_items,
)


def _rival(family: str) -> str:
    f = family.lower()
    if "claude" in f:
        return "GPT-5.2"
    if "gpt" in f or "openai" in f:
        return "Claude Opus 4.5"
    if "qwen" in f:
        return "DeepSeek-V3"
    if "deepseek" in f:
        return "Qwen3"
    return "a competing model"


@dataclass
class RunConfig:
    items_path: str = "data/items.jsonl"
    axes: tuple[str, ...] = (
        "position",
        "length",
        "authority",
        "format",
        "verbose_cot",
        "abstention",
        "self_preference",
    )
    reps: int = 5
    temperature: float = 0.7
    templates: tuple[str, ...] = ("v1_standard",)
    seed: int = 20260830
    output_dir: str = "results"
    run_name: str = ""
    include_noise_floor: bool = True
    resume: bool = True


@dataclass
class Experiment:
    config: RunConfig
    judges: Sequence[Judge]
    items: list[Item] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.items:
            self.items = load_items(self.config.items_path)
        self.rng = random.Random(self.config.seed)
        self._presentations: dict[str, Presentation] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)

    # -- paths ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.run_name or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    @property
    def raw_path(self) -> str:
        return os.path.join(self.config.output_dir, f"{self.name}_raw.jsonl")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.config.output_dir, f"{self.name}_summary.json")

    # -- trial construction --------------------------------------------

    def build_trials(self) -> list[Trial]:
        trials: list[Trial] = []
        for judge in self.judges:
            ctx = {"judge_family": judge.family, "rival_family": _rival(judge.family)}
            for axis_name in self.config.axes:
                axis = get_axis(axis_name)
                for item in self.items:
                    pair = axis.build(item, self.rng, ctx)
                    conditions: list[tuple[Condition, Presentation]] = [
                        (Condition.BASE, pair.base),
                        (Condition.PERTURBED, pair.perturbed),
                    ]
                    if self.config.include_noise_floor:
                        conditions.append((Condition.NOISE_A, pair.base))
                        conditions.append((Condition.NOISE_B, pair.base))
                    for condition, pres in conditions:
                        pkey = f"{item.id}|{axis_name}|{condition.value}|{judge.id}"
                        self._presentations[pkey] = pres
                        for template in self.config.templates:
                            for rep in range(self.config.reps):
                                trials.append(
                                    Trial(
                                        item_id=item.id,
                                        axis=axis_name,
                                        condition=condition,
                                        judge_id=judge.id,
                                        rep=rep,
                                        temperature=self.config.temperature,
                                        prompt_template=template,
                                        preferred=pres.preferred,
                                        blocks=pres.blocks,
                                        extra_instruction=axis.extra_instruction(item, ctx),
                                    )
                                )
        return trials

    # -- execution ------------------------------------------------------

    def _done_keys(self) -> set[str]:
        if not (self.config.resume and os.path.exists(self.raw_path)):
            return set()
        keys: set[str] = set()
        with open(self.raw_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                keys.add(rec["key"])
        return keys

    def run(self, verbose: bool = True) -> list[Judgment]:
        trials = self.build_trials()
        done = self._done_keys()
        pending = [t for t in trials if t.key not in done]
        if verbose:
            print(f"[jrt] {len(trials)} trials, {len(done)} cached, {len(pending)} to run")

        judge_map = {j.id: j for j in self.judges}
        item_map = {i.id: i for i in self.items}
        out: list[Judgment] = []
        # Appending to a file we were told to restart would silently double
        # every trial and inflate the sample size.
        append = self.config.resume and os.path.exists(self.raw_path)
        fh = open(self.raw_path, "a" if append else "w", encoding="utf-8")
        try:
            for n, trial in enumerate(pending, 1):
                judge = judge_map[trial.judge_id]
                item = item_map[trial.item_id]
                pkey = f"{trial.item_id}|{trial.axis}|{trial.condition.value}|{judge.id}"
                pres = self._presentations[pkey]
                comp = judge.judge_presentation(
                    pres,
                    item.question,
                    template=trial.prompt_template,
                    extra_instruction=trial.extra_instruction,
                    temperature=trial.temperature,
                )
                judgment = Judgment(
                    trial=trial,
                    raw=comp.text,
                    verdict=parse_verdict(comp.text),
                    latency_ms=comp.latency_ms,
                    model_version=comp.model_version,
                    error=comp.error,
                )
                rec = judgment.to_record()
                rec["key"] = trial.key
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 25 == 0:
                    fh.flush()
                    if verbose:
                        print(f"  ... {n}/{len(pending)}")
                out.append(judgment)
        finally:
            fh.close()
        if verbose:
            print(f"[jrt] wrote {len(out)} judgments -> {self.raw_path}")
        return out

    # -- analysis -------------------------------------------------------

    def load_judgments(self) -> list[Judgment]:
        out: list[Judgment] = []
        if not os.path.exists(self.raw_path):
            return out
        with open(self.raw_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(_judgment_from_record(json.loads(line)))
        return out

    def analyse(self, judgments: list[Judgment] | None = None) -> dict[str, list[AxisResult]]:
        judgments = judgments if judgments is not None else self.load_judgments()
        by_judge: dict[str, list[AxisResult]] = {}
        for judge in self.judges:
            results: list[AxisResult] = []
            for axis_name in self.config.axes:
                axis = get_axis(axis_name)
                cells = _pair_cells(
                    judgments, judge.id, axis_name, Condition.BASE, Condition.PERTURBED
                )
                noise = (
                    _pair_cells(judgments, judge.id, axis_name, Condition.NOISE_A, Condition.NOISE_B)
                    if self.config.include_noise_floor
                    else []
                )
                if not cells:
                    continue
                results.append(
                    summarise_axis(
                        axis_name,
                        HYPOTHESES[axis_name],
                        cells,
                        noise,
                        confirmatory=axis.confirmatory,
                    )
                )
            by_judge[judge.id] = apply_correction(results)
        return by_judge

    def write_summary(self, analyses: dict[str, list[AxisResult]]) -> str:
        payload = {
            "run": self.name,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "judges": [
                {"id": j.id, "family": j.family, "type": type(j).__name__} for j in self.judges
            ],
            "n_items": len(self.items),
            "results": {
                jid: [_axis_to_dict(r) for r in rs] for jid, rs in analyses.items()
            },
        }
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return self.summary_path


def _pair_cells(
    judgments: Iterable[Judgment],
    judge_id: str,
    axis: str,
    cond_x: Condition,
    cond_y: Condition,
) -> list[PairedOutcome]:
    """Pair verdicts by (item, template, rep) across two conditions."""
    buckets: dict[tuple[str, str, int], dict[Condition, bool]] = {}
    for j in judgments:
        if j.judge_id != judge_id or j.axis != axis or not j.is_scorable:
            continue
        if j.trial.condition not in (cond_x, cond_y):
            continue
        k = (j.trial.item_id, j.trial.prompt_template, j.trial.rep)
        buckets.setdefault(k, {})[j.trial.condition] = j.is_correct
    cells: list[PairedOutcome] = []
    per_item: dict[str, list[int]] = {}
    for (item_id, _tpl, _rep), d in buckets.items():
        if cond_x not in d or cond_y not in d:
            continue
        x_ok, y_ok = d[cond_x], d[cond_y]
        b = 1 if (x_ok and not y_ok) else 0
        c = 1 if (not x_ok and y_ok) else 0
        row = per_item.setdefault(item_id, [0, 0, 0])
        row[0] += 1
        row[1] += b
        row[2] += c
    for item_id, (n, b, c) in per_item.items():
        cells.append(PairedOutcome(n=n, b=b, c=c, item_id=item_id))
    return cells


def _axis_to_dict(r: AxisResult) -> dict:
    d = asdict(r)
    d["h_ci"] = list(r.h_ci)
    d["noise_ci"] = list(r.noise_ci)
    return d


def _judgment_from_record(rec: dict) -> Judgment:
    t = rec
    trial = Trial(
        item_id=t["item_id"],
        axis=t["axis"],
        condition=Condition(t["condition"]),
        judge_id=t["judge_id"],
        rep=t["rep"],
        temperature=t["temperature"],
        prompt_template=t["prompt_template"],
        preferred=t["preferred"],
        blocks=tuple((b["label"], b["text"]) for b in t["blocks"]),
        extra_instruction=t.get("extra_instruction", ""),
    )
    return Judgment(
        trial=trial,
        raw=t["raw"],
        verdict=Verdict(t["verdict"]),
        latency_ms=t["latency_ms"],
        model_version=t.get("model_version", ""),
        error=t.get("error", ""),
    )


__all__ = ["RunConfig", "Experiment", "dump_jsonl"]
