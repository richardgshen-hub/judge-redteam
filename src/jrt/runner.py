"""Experiment orchestration.

Replicates the preregistered protocol: paired base/perturbed conditions, R
replicates at temperature > 0, a noise floor from two unperturbed repeats, full
raw-response persistence, and crash-safe resume.

Run identity
------------
Every run has exactly one identity, computed once at construction time from the
things that determine what the data *means*: the axes, the replicate count, the
temperature, the templates, the seed, the item set's content hash, and the set
of judges. Two invocations of the same command therefore produce the same run
id and the second one resumes the first. Change anything that would invalidate
comparison and you get a different id rather than a silently mixed dataset.

This matters because a long run against a paid API is exactly the situation
where a wall-clock-derived name is most dangerous: `raw`, `summary`, and
`report` are computed at different moments, so a run that crosses a second
boundary writes its evidence to one path and its conclusions to another.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

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

MANIFEST_SCHEMA_VERSION = 2


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


def sha256_file(path: str) -> str:
    """Full content hash of the item set, so a run can be tied to exact data."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    """Short SHA of the checkout, or 'unknown' outside a repo.

    Never fatal: a run inside a wheel or a tarball has no commit and that is
    fine, it just means the provenance record is weaker.
    """
    try:
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def _fingerprint(config: "RunConfig", items_hash: str, judges: Sequence[Judge]) -> str:
    """Identity of the experiment: everything that changes what the data means.

    Deliberately excludes output_dir and run_name, which say where results go
    rather than what they are, and excludes wall-clock time, which makes
    identity unstable by construction.
    """
    payload = json.dumps(
        {
            "schema": MANIFEST_SCHEMA_VERSION,
            "axes": sorted(config.axes),
            "reps": config.reps,
            "temperature": config.temperature,
            "templates": sorted(config.templates),
            "seed": config.seed,
            "noise_floor": config.include_noise_floor,
            "items_sha256": items_hash,
            "judges": sorted(j.id for j in judges),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class ConfigConflict(RuntimeError):
    """Raised when a resume would mix data from two different experiments."""


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

    def as_record(self) -> dict[str, Any]:
        # Canonical, order-independent form: the identity fingerprint sorts axes
        # and templates, so the recorded config must too. Two runs that differ
        # only in axis ordering are the same experiment and must compare equal.
        d = asdict(self)
        d["axes"] = sorted(self.axes)
        d["templates"] = sorted(self.templates)
        return d


@dataclass
class RunManifest:
    """What was actually run, recorded before any data is collected.

    The point is that a partially finished run can be compared against a new
    invocation and either resumed or rejected. Silent mixing is the failure
    mode this exists to prevent.
    """

    run_id: str
    schema: int
    created_utc: str
    config: dict[str, Any]
    items_path: str
    items_sha256: str
    items_count: int
    judges: list[dict[str, str]]
    code_commit: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_record(rec: dict[str, Any]) -> "RunManifest":
        return RunManifest(
            run_id=rec.get("run_id", ""),
            schema=int(rec.get("schema", 0)),
            created_utc=rec.get("created_utc", ""),
            config=rec.get("config", {}),
            items_path=rec.get("items_path", ""),
            items_sha256=rec.get("items_sha256", ""),
            items_count=int(rec.get("items_count", 0)),
            judges=rec.get("judges", []),
            code_commit=rec.get("code_commit", "unknown"),
        )

    def diff(self, other: "RunManifest") -> list[str]:
        """Human-readable list of everything that differs, for a refusal message."""
        out = []
        if self.items_sha256 != other.items_sha256:
            out.append(
                f"item set: {self.items_sha256[:12]} vs {other.items_sha256[:12]}"
            )
        if self.config != other.config:
            for k in sorted(set(self.config) | set(other.config)):
                a, b = self.config.get(k), other.config.get(k)
                if a != b:
                    out.append(f"{k}: {a!r} vs {b!r}")
        if sorted(j["id"] for j in self.judges) != sorted(j["id"] for j in other.judges):
            out.append(
                "judges: "
                f"{sorted(j['id'] for j in self.judges)} vs "
                f"{sorted(j['id'] for j in other.judges)}"
            )
        return out


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

        self.items_sha256 = sha256_file(self.config.items_path)
        self.code_commit_sha = code_commit()
        # Computed once. Everything downstream reads this attribute; nothing
        # recomputes it. A run that spans an hour still has one identity.
        self._run_id = self.config.run_name or _fingerprint(
            self.config, self.items_sha256, self.judges
        )
        self.manifest = RunManifest(
            run_id=self._run_id,
            schema=MANIFEST_SCHEMA_VERSION,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            config=self.config.as_record(),
            items_path=self.config.items_path,
            items_sha256=self.items_sha256,
            items_count=len(self.items),
            judges=[
                {"id": j.id, "family": j.family, "type": type(j).__name__} for j in self.judges
            ],
            code_commit=self.code_commit_sha,
        )

    # -- paths ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._run_id

    @property
    def raw_path(self) -> str:
        return os.path.join(self.config.output_dir, f"{self.name}_raw.jsonl")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.config.output_dir, f"{self.name}_summary.json")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.config.output_dir, f"{self.name}_manifest.json")

    # -- manifest / resume --------------------------------------------

    def write_manifest(self) -> str:
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest.to_record(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return self.manifest_path

    def load_manifest(self) -> RunManifest | None:
        if not os.path.exists(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                return RunManifest.from_record(json.load(fh))
        except (OSError, ValueError):
            return None

    def check_resumable(self) -> tuple[bool, str]:
        """Decide whether existing raw data may be appended to.

        Refusing is cheap; silent mixing produces a dataset that no longer
        answers any question and looks fine while doing it.
        """
        if not os.path.exists(self.raw_path):
            return True, "no existing data"
        if not self.config.resume:
            return True, "resume disabled: existing data will be replaced"
        prior = self.load_manifest()
        if prior is None:
            return False, (
                f"refusing to resume {self.raw_path}: no manifest found.\n"
                "Data written by an older version cannot be verified as compatible. "
                "Move it aside or pass --no-resume."
            )
        diffs = prior.diff(self.manifest)
        if diffs:
            return False, (
                f"refusing to resume run {self.name}: the existing data came from a "
                "different configuration.\n"
                + "\n".join(f"  - {d}" for d in diffs)
                + "\nUse a different --name, or --no-resume to start over."
            )
        return True, "configuration matches"

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

    def expected_calls(self, trials: Sequence[Trial] | None = None) -> int:
        """Total judge invocations a full run will issue, for the cost preview."""
        return len(trials) if trials is not None else len(self.build_trials())

    # -- execution ------------------------------------------------------

    def _iter_records(self, path: str) -> Iterable[dict[str, Any]]:
        """Yield records, tolerating a truncated final line.

        A run killed mid-write leaves a partial last line. Dropping it is safe
        because that trial was never acknowledged, so a resume will re-issue it.
        """
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _done_keys(self) -> set[str]:
        if not (self.config.resume and os.path.exists(self.raw_path)):
            return set()
        return {rec["key"] for rec in self._iter_records(self.raw_path) if "key" in rec}

    def run(self, verbose: bool = True) -> list[Judgment]:
        ok, reason = self.check_resumable()
        if not ok:
            raise ConfigConflict(reason)
        if verbose and reason != "no existing data":
            print(f"[jrt] resume check: {reason}")

        trials = self.build_trials()
        done = self._done_keys()
        pending = [t for t in trials if t.key not in done]
        if verbose:
            print(f"[jrt] run {self.name}: {len(trials)} trials, "
                  f"{len(done)} cached, {len(pending)} to run")

        self.write_manifest()

        judge_map = {j.id: j for j in self.judges}
        item_map = {i.id: i for i in self.items}
        out: list[Judgment] = []
        # Appending to a file we were told to restart would silently double
        # every trial and inflate the sample size.
        append = self.config.resume and os.path.exists(self.raw_path)
        fh = open(self.raw_path, "a" if append else "w", encoding="utf-8")
        interrupted = False

        def _stop(signum, _frame):
            nonlocal interrupted
            interrupted = True
            if verbose:
                print("\n[jrt] interrupt received, flushing...", flush=True)

        prev = signal.signal(signal.SIGINT, _stop)
        try:
            for n, trial in enumerate(pending, 1):
                if interrupted:
                    break
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
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
            signal.signal(signal.SIGINT, prev)

        if verbose:
            if interrupted:
                print(
                    f"[jrt] interrupted after {len(out)} judgments; "
                    f"{len(pending) - len(out)} remain. Re-run the same command to resume."
                )
            else:
                print(f"[jrt] wrote {len(out)} judgments -> {self.raw_path}")
        return out

    # -- analysis -------------------------------------------------------

    def load_judgments(self) -> list[Judgment]:
        """Load recorded judgments, de-duplicated by trial key.

        De-duplication is a safety net rather than the primary mechanism: a
        trial repeated by an older build would otherwise be counted twice and
        quietly inflate every reported sample size.
        """
        out: list[Judgment] = []
        if not os.path.exists(self.raw_path):
            return out
        seen: set[str] = set()
        for rec in self._iter_records(self.raw_path):
            key = rec.get("key", "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out.append(_judgment_from_record(rec))
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
            "run_id": self.name,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": self.manifest.to_record(),
            "config": self.config.as_record(),
            "judges": [
                {"id": j.id, "family": j.family, "type": type(j).__name__} for j in self.judges
            ],
            "n_items": len(self.items),
            "results": {jid: [_axis_to_dict(r) for r in rs] for jid, rs in analyses.items()},
        }
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return self.summary_path


def _pair_cells(
    judgments: Iterable[Judgment],
    judge_id: str,
    axis: str,
    cond_x: Condition,
    cond_y: Condition,
) -> list[PairedOutcome]:
    """Pair verdicts by (item, template, rep) across two conditions.

    One PairedOutcome per item: all of that item's replicates and templates are
    pooled into a single cluster. Downstream inference resamples items, never
    individual verdicts, because ten verdicts about the same question are ten
    correlated observations of one thing, not ten independent facts.
    """
    buckets: dict[tuple[str, str, int], dict[Condition, bool]] = {}
    for j in judgments:
        if j.judge_id != judge_id or j.axis != axis or not j.is_scorable:
            continue
        if j.trial.condition not in (cond_x, cond_y):
            continue
        k = (j.trial.item_id, j.trial.prompt_template, j.trial.rep)
        buckets.setdefault(k, {})[j.trial.condition] = j.is_correct
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
    return [PairedOutcome(n=n, b=b, c=c, item_id=iid) for iid, (n, b, c) in per_item.items()]


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


__all__ = [
    "RunConfig",
    "RunManifest",
    "Experiment",
    "ConfigConflict",
    "sha256_file",
    "code_commit",
    "dump_jsonl",
]
