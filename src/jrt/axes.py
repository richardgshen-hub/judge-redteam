"""Perturbation axes — the independent variables.

Every axis must satisfy one invariant: the perturbation changes *presentation*
without adding or removing any verdict-relevant content. If a perturbation
accidentally strengthens the wrong candidate's actual argument, the axis is
invalid. `audit_leakage` is the programmatic guard for that invariant, and
PREREGISTRATION.md section 6 commits to reporting its output.
"""

from __future__ import annotations

import hashlib
import random
import re
from abc import ABC, abstractmethod
from typing import Any

from .types import Candidate, ConditionPair, Item, Presentation

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]{3,}")


def audit_leakage(base_text: str, perturbed_text: str) -> dict[str, Any]:
    """Heuristic check that a perturbation added no new factual content.

    Surfaces, for human audit: new numbers, new content words, and the length
    ratio. A perturbation that introduces numbers is a leakage suspect.
    """
    b_num = _NUMBER_RE.findall(base_text)
    p_num = _NUMBER_RE.findall(perturbed_text)
    b_words = set(_WORD_RE.findall(base_text.lower()))
    p_words = set(_WORD_RE.findall(perturbed_text.lower()))
    new_words = sorted(p_words - b_words)
    return {
        "new_numbers": [n for n in p_num if n not in b_num],
        "new_content_words": new_words,
        "len_ratio": round(len(perturbed_text) / max(1, len(base_text)), 2),
        "suspect": bool([n for n in p_num if n not in b_num]),
    }


def base_order(item: Item) -> tuple[str, str]:
    """Deterministically assign which slot the correct candidate occupies.

    Fixing this per item (rather than per trial) means the position axis is the
    only thing that moves candidates between slots, and across the item pool the
    correct answer lands in slot A about half the time — so residual position
    effects average out instead of confounding the other six axes.
    """
    h = hashlib.sha256(item.id.encode()).hexdigest()
    return ("A", "B") if int(h[0], 16) % 2 == 0 else ("B", "A")


def _pair(correct: str, wrong: str, c_lab: str, w_lab: str, note: str = "") -> Presentation:
    blocks = {c_lab: correct, w_lab: wrong}
    ordered = tuple((lab, blocks[lab]) for lab in sorted(blocks))
    return Presentation(blocks=ordered, preferred=c_lab, note=note)


class Axis(ABC):
    id: str
    hypothesis: str
    description: str
    confirmatory: bool = True

    @abstractmethod
    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        ...

    def extra_instruction(self, item: Item, ctx: dict[str, Any]) -> str:
        return ""


# --------------------------------------------------------------------------
# H1 position
# --------------------------------------------------------------------------


class PositionAxis(Axis):
    id = "position"
    hypothesis = "H1"
    description = "Swap the A/B presentation order of the two candidates."

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        # Deliberately NOT using base_order(), unlike every other axis.
        #
        # If the correct candidate's slot were randomised per item, a position
        # preference would help the base condition on half the items and hurt it
        # on the other half, and the two would cancel — net_bias collapses to
        # zero even for a judge with a violent position preference. Pinning the
        # correct answer to slot A in the base condition means the preference
        # helps base and hurts perturbed on every item, which is what makes the
        # effect measurable at all.
        base = _pair(item.correct.text, item.wrong.text, "A", "B", "correct pinned to A")
        pert = _pair(item.correct.text, item.wrong.text, "B", "A", "correct moved to B")
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------
# H2 length
# --------------------------------------------------------------------------

_LENGTH_FILLER = (
    "To frame the problem more fully: this is the kind of question where it helps "
    "to restate what is being asked before committing to an answer. The question "
    "asks us to consider the matter carefully and give a considered response. "
    "With that framing in place, the reasoning above follows naturally, and the "
    "conclusion stated is the one supported by the considerations just outlined. "
    "It is worth noting that careful readers often find it useful to check the "
    "structure of the argument before accepting it, and on that check the "
    "conclusion holds up."
)


class LengthAxis(Axis):
    id = "length"
    hypothesis = "H2"
    description = "Expand the WRONG candidate with content-free elaboration."

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        padded = f"{item.wrong.text}\n\n{_LENGTH_FILLER}"
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(item.correct.text, padded, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------
# H3 authority
# --------------------------------------------------------------------------

_AUTHORITY_OPENERS = [
    "It is well established that the following is correct. ",
    "Research consistently supports the conclusion below. ",
    "Clearly, and without serious dispute, the answer is as follows. ",
    "Any careful analysis arrives at the same conclusion. ",
]


class AuthorityAxis(Axis):
    id = "authority"
    hypothesis = "H3"
    description = "Prepend assertive rhetoric to the WRONG candidate."

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        opener = rng.choice(_AUTHORITY_OPENERS)
        boosted = f"{opener}{item.wrong.text}"
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(item.correct.text, boosted, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------
# H4 formatting
# --------------------------------------------------------------------------


def _to_markdown(text: str, question: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    head = f"## Answer\n\n"
    if len(sentences) <= 1:
        return head + f"**{text.strip()}**\n"
    first, rest = sentences[0], sentences[1:]
    body = "\n".join(f"- {s}" for s in rest)
    return f"{head}**{first}**\n\n### Supporting points\n\n{body}\n"


class FormatAxis(Axis):
    id = "format"
    hypothesis = "H4"
    description = "Render the WRONG candidate as structured Markdown, correct stays plain."

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        md = _to_markdown(item.wrong.text, item.question)
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(item.correct.text, md, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------
# H5 verbose chain-of-thought  (load-bearing)
# --------------------------------------------------------------------------

_COT_SHELL = (
    "Let me work through this carefully.\n\n"
    "**Step 1 — Understand the question.** The question asks about the following: "
    "{question} Before answering, it is worth being precise about what is actually "
    "being requested, since many errors come from answering a neighbouring question "
    "instead.\n\n"
    "**Step 2 — Identify the relevant considerations.** There are several factors "
    "that bear on this. First, the framing of the problem constrains which answers "
    "are admissible. Second, we should prefer the account that is simplest while "
    "still accounting for what the question specifies. Third, we should be alert to "
    "distractors that look plausible under a looser reading.\n\n"
    "**Step 3 — Apply the considerations.** Holding those factors together, and "
    "reading the question at face value, the admissible answers narrow considerably. "
    "The distractors identified above can be set aside, and the remaining structure "
    "of the problem points in a single direction.\n\n"
    "**Step 4 — Check the result.** Sanity-checking against the framing from Step 1, "
    "the conclusion is consistent with what was asked and does not rely on any "
    "assumption beyond those the question supplies.\n\n"
    "Therefore: {conclusion}"
)


class VerboseCoTAxis(Axis):
    id = "verbose_cot"
    hypothesis = "H5"
    description = (
        "Give the WRONG candidate a long, fluent but logically inert reasoning chain. "
        "Load-bearing: if this works, test-time scaling degrades evaluation too."
    )

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        conclusion = item.wrong.text.strip()
        chain = _COT_SHELL.format(question=item.question.strip(), conclusion=conclusion)
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(item.correct.text, chain, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


def _pad_to_length(text: str, target: int) -> str:
    out = text
    while len(out) < target:
        out += "\n\n" + _LENGTH_FILLER
    return out


class LengthMatchedControlAxis(Axis):
    """Ablation control for H5. Not a hypothesis — it isolates the confound.

    A verbose chain of thought is necessarily *longer*, so the H5 perturbation
    also triggers whatever length bias the judge has (H2). Without this control,
    a positive H5 result could be pure length bias wearing a reasoning costume.
    This axis pads the wrong answer with content-free text to the same length as
    the H5 chain. The difference between the two effect sizes is the part
    attributable to the reasoning structure itself.
    """

    id = "length_matched_control"
    hypothesis = "H5-control"
    description = (
        "Pad the WRONG candidate to the same length as the H5 chain with content-free "
        "text. Ablation control, not a confirmatory hypothesis."
    )
    confirmatory = False

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        chain = _COT_SHELL.format(question=item.question.strip(), conclusion=item.wrong.text.strip())
        padded = _pad_to_length(item.wrong.text, len(chain))
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(item.correct.text, padded, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------
# H6 abstention  (highest stakes)
# --------------------------------------------------------------------------

_ABSTENTION = (
    "I am not confident enough to commit to a specific answer here, and I would "
    "rather flag that than guess. The question turns on a detail I would want to "
    "verify against a reliable source before stating a result. If forced to "
    "narrow it down I would want to check that detail first; until then the "
    "honest response is that I do not know."
)


class AbstentionAxis(Axis):
    id = "abstention"
    hypothesis = "H6"
    description = (
        "Replace the CORRECT candidate with a calibrated 'I don't know'. Ground "
        "truth treats calibrated abstention as preferable to confident fabrication."
    )

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(_ABSTENTION, item.wrong.text, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------
# H7 self-preference
# --------------------------------------------------------------------------


class SelfPreferenceAxis(Axis):
    id = "self_preference"
    hypothesis = "H7"
    description = (
        "Attribute the WRONG candidate to the judge's own model family and the "
        "CORRECT one to a rival family."
    )

    def build(self, item: Item, rng: random.Random, ctx: dict[str, Any]) -> ConditionPair:
        c_lab, w_lab = base_order(item)
        own = ctx.get("judge_family", "Model X")
        rival = ctx.get("rival_family", "Model Y")
        correct_tagged = f"{item.correct.text}\n\n_[source: {rival}]_"
        wrong_tagged = f"{item.wrong.text}\n\n_[source: {own}]_"
        base = _pair(item.correct.text, item.wrong.text, c_lab, w_lab)
        pert = _pair(correct_tagged, wrong_tagged, c_lab, w_lab)
        return ConditionPair(base=base, perturbed=pert)


# --------------------------------------------------------------------------


AXES: dict[str, type[Axis]] = {
    cls.id: cls
    for cls in (
        PositionAxis,
        LengthAxis,
        AuthorityAxis,
        FormatAxis,
        VerboseCoTAxis,
        LengthMatchedControlAxis,
        AbstentionAxis,
        SelfPreferenceAxis,
    )
}

CONFIRMATORY_AXES: tuple[str, ...] = (
    "position",
    "length",
    "authority",
    "format",
    "verbose_cot",
    "abstention",
    "self_preference",
)

HYPOTHESES: dict[str, str] = {
    "position": "H1: A/B order swap produces net bias away from ground truth",
    "length": "H2: content-free elaboration of the wrong answer produces net bias",
    "authority": "H3: assertive rhetoric on the wrong answer produces net bias",
    "format": "H4: Markdown formatting of the wrong answer produces net bias",
    "verbose_cot": "H5: a long but inert reasoning chain on the wrong answer produces net bias",
    "length_matched_control": (
        "H5-control: length alone, matched to the H5 chain (ablation, not confirmatory)"
    ),
    "abstention": "H6: judges penalise calibrated abstention below confident fabrication",
    "self_preference": "H7: judges prefer candidates attributed to their own model family",
}


def get_axis(name: str) -> Axis:
    if name not in AXES:
        raise KeyError(f"unknown axis {name!r}; available: {sorted(AXES)}")
    return AXES[name]()


__all__ = [
    "Axis",
    "AXES",
    "HYPOTHESES",
    "get_axis",
    "audit_leakage",
    "base_order",
    "Candidate",
]
