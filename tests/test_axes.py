"""Tests for the perturbation axes: taxonomy, integrity, and framing.

Problem 4 (research concept & naming). H6 is a behavioral intervention (abstention
robustness), not a pure surface-form perturbation; H7 is an attribution /
identity-label bias (metadata), not true self-preference. These tests lock in that
classification and the invariant that no axis silently adds verdict-relevant content.
"""

from __future__ import annotations

import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jrt import axes  # noqa: E402
from jrt.types import Candidate, Domain, Item  # noqa: E402

_ITEM = Item(
    id="t1",
    domain=Domain.ARITHMETIC,
    question="What is 2+2?",
    correct=Candidate(text="4", is_correct=True),
    wrong=Candidate(text="5", is_correct=False),
)


def _build(axis_id):
    axis = axes.get_axis(axis_id)
    rng = random.Random(1)
    ctx = {"judge_family": "GPT-5.2", "rival_family": "Claude Opus 4.5"}
    return axis, axis.build(_ITEM, rng, ctx)


def test_confirmatory_family_has_seven_axes():
    assert len(axes.CONFIRMATORY_AXES) == 7
    assert "length_matched_control" not in axes.CONFIRMATORY_AXES


def test_taxonomy_classifies_h6_h7_correctly():
    # H1-H5 + control are surface-form / presentation; H6 is behavioral; H7 is metadata.
    for ax in ("position", "length", "authority", "format", "verbose_cot"):
        assert axes.get_axis(ax).category == "surface-form"
    assert axes.get_axis("abstention").category == "behavioral"
    assert axes.get_axis("self_preference").category == "metadata"
    assert axes.AXIS_TAXONOMY["abstention"] == "behavioral"
    assert axes.AXIS_TAXONOMY["self_preference"] == "metadata"


def test_h6_description_reflects_behavioral_intervention():
    assert "behavioral" in axes.HYPOTHESES["abstention"].lower()
    assert "abstention robustness" in axes.HYPOTHESES["abstention"].lower()


def test_h7_description_reflects_attribution_bias():
    hyp = axes.HYPOTHESES["self_preference"].lower()
    assert "attribution" in hyp
    assert "self-preference" in hyp  # explicitly named as NOT true self-preference


_BENIGN_NUMBERS = {
    # verbose_cot's inert shell uses structural step numbering ("Step 1".."Step 4").
    # Those are scaffolding, not factual content, so they are whitelisted per axis.
    "verbose_cot": {"1", "2", "3", "4"},
}

_SOURCE_TAG = re.compile(r"_\[source:[^\]]*\]_")
# For H7 the [source: ...] provenance tag IS the manipulation (a metadata axis),
# so it is stripped before the leakage audit: its version numbers are the
# treatment variable, not smuggled factual content.
_STRIP_TREATMENT = {"self_preference": _SOURCE_TAG.sub}


def test_no_axis_introduces_new_numbers():
    # The leakage invariant: a perturbation must not inject new FACTUAL content.
    # Compare the WHOLE presentation (all blocks), not slot A alone — the position
    # axis swaps slots, so slot A alone would flag the swapped answer as "new".
    # audit_leakage is a heuristic that surfaces suspects for human review; this
    # test enforces the hard part (numbers) with a per-axis structural whitelist.
    for axis_id in axes.AXES:
        _axis, pair = _build(axis_id)
        base_all = "\n".join(text for _label, text in pair.base.blocks)
        pert_all = "\n".join(text for _label, text in pair.perturbed.blocks)
        strip = _STRIP_TREATMENT.get(axis_id)
        if strip is not None:
            base_all = strip("", base_all)
            pert_all = strip("", pert_all)
        leak = axes.audit_leakage(base_all, pert_all)
        benign = _BENIGN_NUMBERS.get(axis_id, set())
        extra = [n for n in leak["new_numbers"] if n not in benign]
        assert not extra, f"{axis_id} introduced non-whitelisted numbers: {extra}"


def test_h7_tags_both_candidates_with_source_label():
    _axis, pair = _build("self_preference")
    correct_text = pair.perturbed.blocks[0][1]
    wrong_text = pair.perturbed.blocks[1][1]
    assert "[source:" in correct_text and "[source:" in wrong_text
    # correct is tagged with the RIVAL family, wrong with the OWN family: that is
    # what makes it an attribution test, not a content change.
    assert "Claude Opus 4.5" in correct_text
    assert "GPT-5.2" in wrong_text


def test_h6_replaces_correct_with_abstention():
    _axis, pair = _build("abstention")
    correct_text = pair.perturbed.blocks[0][1]
    assert "do not know" in correct_text.lower()
