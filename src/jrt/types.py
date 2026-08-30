"""Core data structures for judge-redteam.

Design note: ground truth is carried per-presentation, not per-candidate, because
hypothesis H6 (abstention) deliberately changes which surface form *should* win
without changing which side is correct. See PREREGISTRATION.md section 3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class Domain(str, Enum):
    """Item domains restricted to those with decidable correctness."""

    ARITHMETIC = "arithmetic"
    LOGIC = "logic"
    CODE = "code"
    FACT = "fact"
    CONSTRAINT = "constraint"


class Verdict(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"
    PARSE_FAIL = "PARSE_FAIL"


class Condition(str, Enum):
    BASE = "base"
    PERTURBED = "perturbed"
    NOISE_A = "noise_a"
    NOISE_B = "noise_b"


@dataclass(frozen=True)
class Candidate:
    """One answer to an item. `is_correct` is the domain ground truth."""

    text: str
    is_correct: bool
    tag: str = ""

    def fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Item:
    """A question with one correct and one plausible-but-wrong answer."""

    id: str
    domain: Domain
    question: str
    correct: Candidate
    wrong: Candidate
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.correct.is_correct or self.wrong.is_correct:
            raise ValueError(f"item {self.id}: candidate correctness flags are inconsistent")


@dataclass(frozen=True)
class Presentation:
    """What the judge actually sees for one condition.

    `preferred` names the label that *should* win under ground truth. It is
    per-presentation rather than per-candidate so that H6 can swap the winning
    surface form (a real answer vs. a calibrated abstention) without breaking
    the correctness bookkeeping.
    """

    blocks: tuple[tuple[str, str], ...]
    preferred: str
    note: str = ""

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.blocks)


@dataclass(frozen=True)
class ConditionPair:
    """The base and perturbed presentations for one (item, axis) cell."""

    base: Presentation
    perturbed: Presentation


@dataclass(frozen=True)
class Trial:
    """A single judge invocation to be issued."""

    item_id: str
    axis: str
    condition: Condition
    judge_id: str
    rep: int
    temperature: float
    prompt_template: str
    preferred: str
    blocks: tuple[tuple[str, str], ...]
    extra_instruction: str = ""

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.item_id,
                self.axis,
                self.condition.value,
                self.judge_id,
                self.prompt_template,
                str(self.rep),
            )
        )


@dataclass
class Judgment:
    """One raw judge response plus its parse."""

    trial: Trial
    raw: str
    verdict: Verdict
    latency_ms: float
    model_version: str = ""
    error: str = ""

    @property
    def judge_id(self) -> str:
        return self.trial.judge_id

    @property
    def axis(self) -> str:
        return self.trial.axis

    @property
    def condition(self) -> Condition:
        return self.trial.condition

    @property
    def is_correct(self) -> bool:
        return self.verdict.value == self.trial.preferred

    @property
    def is_scorable(self) -> bool:
        return self.verdict in (Verdict.A, Verdict.B)

    def to_record(self) -> dict[str, Any]:
        d = asdict(self.trial)
        d["blocks"] = [{"label": l, "text": t} for l, t in self.trial.blocks]
        d.update(
            {
                "raw": self.raw,
                "verdict": self.verdict.value,
                "is_correct": self.is_correct,
                "latency_ms": round(self.latency_ms, 2),
                "model_version": self.model_version,
                "error": self.error,
            }
        )
        return d


@dataclass
class PairedOutcome:
    """Direction-decomposed flip counts for one (item, axis, judge) cell.

    b = base correct and perturbed wrong  (the perturbation broke the judge)
    c = base wrong   and perturbed correct (the perturbation accidentally helped)
    """

    n: int
    b: int
    c: int
    item_id: str = ""

    @property
    def net_bias(self) -> float:
        return (self.b - self.c) / self.n if self.n else 0.0

    @property
    def flip_rate(self) -> float:
        return (self.b + self.c) / self.n if self.n else 0.0

    @property
    def p_flip_wrong(self) -> float:
        return self.b / self.n if self.n else 0.0

    @property
    def p_flip_right(self) -> float:
        return self.c / self.n if self.n else 0.0


def load_items(path: str) -> list[Item]:
    items: list[Item] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            items.append(
                Item(
                    id=raw["id"],
                    domain=Domain(raw["domain"]),
                    question=raw["question"],
                    correct=Candidate(text=raw["correct"], is_correct=True),
                    wrong=Candidate(text=raw["wrong"], is_correct=False),
                    notes=raw.get("notes", ""),
                )
            )
    return items


def dump_jsonl(records: Iterable[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


__all__ = [
    "Domain",
    "Verdict",
    "Condition",
    "Candidate",
    "Item",
    "Presentation",
    "ConditionPair",
    "Trial",
    "Judgment",
    "PairedOutcome",
    "load_items",
    "dump_jsonl",
]
