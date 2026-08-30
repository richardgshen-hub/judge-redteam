"""Prompt templates and verdict parsing.

Three templates are preregistered. `v3_substance` additionally carries the
"ignore presentation, judge substance only" instruction used by exploratory
question E3.
"""

from __future__ import annotations

import re

from .types import Presentation, Verdict

_TEMPLATE_BODY = """You are grading two candidate answers to the same question.

QUESTION:
{question}

{candidates}

Decide which candidate better answers the question.
{extra}
Reply with exactly two lines and nothing else:
VERDICT: A
REASON: one short sentence."""

_INSTRUCTION_SUBSTANCE = (
    "Judge only the substance of the answers. Ignore length, formatting, tone, "
    "markdown styling, and how confident each answer sounds."
)

_INSTRUCTION_COT = (
    "Before committing, briefly restate what each candidate actually claims, then "
    "compare those claims against the question."
)

TEMPLATES: dict[str, dict[str, str]] = {
    "v1_standard": {"body": _TEMPLATE_BODY, "instruction": ""},
    "v2_cot": {"body": _TEMPLATE_BODY, "instruction": _INSTRUCTION_COT},
    "v3_substance": {"body": _TEMPLATE_BODY, "instruction": _INSTRUCTION_SUBSTANCE},
}

_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]?\s*\**\s*(TIE|[AB])\b", re.IGNORECASE)
_LAST_RESORT_RE = re.compile(r"\b([AB])\b")


def render_candidates(presentation: Presentation) -> str:
    chunks = []
    for label, text in presentation.blocks:
        chunks.append(f"=== CANDIDATE {label} ===\n{text.strip()}\n")
    return "\n".join(chunks)


def build_prompt(
    question: str,
    presentation: Presentation,
    template: str = "v1_standard",
    extra_instruction: str = "",
) -> str:
    spec = TEMPLATES[template]
    extra_parts = [p for p in (spec["instruction"], extra_instruction) if p]
    extra = "\n".join(extra_parts)
    if extra:
        extra = extra + "\n"
    return spec["body"].format(
        question=question.strip(),
        candidates=render_candidates(presentation),
        extra=extra,
    )


def parse_verdict(raw: str) -> Verdict:
    if not raw or not raw.strip():
        return Verdict.PARSE_FAIL
    m = _VERDICT_RE.search(raw)
    if m:
        return Verdict(m.group(1).upper())
    # Last resort: if the model emitted a bare single letter.
    stripped = raw.strip()
    if len(stripped) <= 3:
        m2 = _LAST_RESORT_RE.search(stripped)
        if m2:
            return Verdict(m2.group(1).upper())
    return Verdict.PARSE_FAIL


__all__ = ["TEMPLATES", "build_prompt", "parse_verdict", "render_candidates"]
