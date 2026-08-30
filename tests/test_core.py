"""Tests for the parts that must be right: statistics and perturbation invariants.

These run without network or model access. If the statistics are wrong, every
conclusion downstream is decoration.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jrt.axes import AXES, CONFIRMATORY_AXES, HYPOTHESES, audit_leakage, base_order, get_axis  # noqa: E402
from jrt.prompts import build_prompt, parse_verdict  # noqa: E402
from jrt.stats import (  # noqa: E402
    cluster_bootstrap_ci,
    cohens_h,
    dersimonian_laird,
    holm_bonferroni,
    mcnemar_exact,
)
from jrt.types import Candidate, Item, PairedOutcome, Verdict, load_items  # noqa: E402

ITEMS_PATH = os.path.join(ROOT, "data", "items.jsonl")


@pytest.fixture(scope="module")
def items():
    return load_items(ITEMS_PATH)


@pytest.fixture()
def sample_item():
    return Item(
        id="test01",
        domain="arithmetic",
        question="What is 12 x 11?",
        correct=Candidate("12 x 11 = 132.", True),
        wrong=Candidate("12 x 11 = 121.", False),
    )


# ---------------------------------------------------------------- statistics


def test_mcnemar_symmetric_is_not_significant():
    assert mcnemar_exact(15, 15) > 0.5


def test_mcnemar_asymmetric_is_significant():
    assert mcnemar_exact(25, 5) < 0.001


def test_mcnemar_no_discordance():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_never_exceeds_one():
    for b, c in ((50, 1), (1, 50), (0, 30), (30, 0)):
        assert 0.0 <= mcnemar_exact(b, c) <= 1.0


def test_cohens_h_properties():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.7, 0.3) > 0
    assert cohens_h(0.3, 0.7) == pytest.approx(-cohens_h(0.7, 0.3))


def test_holm_is_monotone_and_capped():
    rejected, adjusted = holm_bonferroni([0.001, 0.02, 0.4], alpha=0.05)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert all(a <= 1.0 for a in adjusted)
    assert rejected[0] is True


def test_holm_rejects_nothing_when_all_weak():
    rejected, _ = holm_bonferroni([0.4, 0.5], alpha=0.05)
    assert rejected == [False, False]


def test_bootstrap_ci_brackets_point_estimate():
    clusters = [[1.0, 1.0, -1.0] for _ in range(30)]
    lo, hi = cluster_bootstrap_ci(clusters, lambda f: sum(f) / len(f), n_resample=2000)
    assert lo <= 1 / 3 <= hi


def test_meta_analysis_recovers_known_effect():
    effects = [0.5] * 20
    variances = [0.05] * 20
    m = dersimonian_laird(effects, variances)
    assert m.pooled == pytest.approx(0.5, abs=1e-6)
    assert m.p < 0.001


def test_net_bias_decomposition():
    o = PairedOutcome(n=100, b=30, c=10)
    assert o.net_bias == pytest.approx(0.20)
    assert o.flip_rate == pytest.approx(0.40)
    assert o.p_flip_wrong == pytest.approx(0.30)
    assert o.p_flip_right == pytest.approx(0.10)


def test_pure_noise_gives_zero_net_bias():
    """The claim the whole method rests on: symmetric flips cancel."""
    o = PairedOutcome(n=100, b=18, c=18)
    assert o.net_bias == 0.0
    assert o.flip_rate > 0.0


# ------------------------------------------------------------- perturbations


def test_every_axis_has_a_hypothesis():
    for name in AXES:
        assert name in HYPOTHESES


def test_confirmatory_family_has_seven():
    assert len(CONFIRMATORY_AXES) == 7
    assert set(CONFIRMATORY_AXES) <= set(AXES)


def test_axes_produce_two_conditions(sample_item):
    rng = random.Random(0)
    ctx = {"judge_family": "Simulated", "rival_family": "GPT-5.2"}
    for name in AXES:
        pair = get_axis(name).build(sample_item, rng, ctx)
        assert len(pair.base.blocks) == 2
        assert len(pair.perturbed.blocks) == 2
        assert pair.base.preferred in ("A", "B")
        assert pair.perturbed.preferred in ("A", "B")


def test_position_axis_swaps_slots(sample_item):
    """H1 must pin, not randomise — see PREREGISTRATION section 3."""
    pair = get_axis("position").build(sample_item, random.Random(0), {})
    assert pair.base.preferred == "A"
    assert pair.perturbed.preferred == "B"


def test_position_axis_is_pinned_across_items(items):
    """Pinning holds for every item, not just this one."""
    for it in items:
        pair = get_axis("position").build(it, random.Random(0), {})
        assert pair.base.preferred == "A"
        assert pair.perturbed.preferred == "B"


@pytest.mark.parametrize("axis_name", ["length", "authority", "format", "length_matched_control"])
def test_perturbation_adds_no_numbers(axis_name, sample_item):
    """Leakage guard: content-free padding must stay content-free."""
    pair = get_axis(axis_name).build(sample_item, random.Random(0), {})
    base_wrong = dict(pair.base.blocks)[
        "A" if pair.base.preferred == "B" else "B"
    ]
    pert_wrong = dict(pair.perturbed.blocks)[
        "A" if pair.perturbed.preferred == "B" else "B"
    ]
    report = audit_leakage(base_wrong, pert_wrong)
    assert report["suspect"] is False, report["new_numbers"]
    assert report["len_ratio"] > 1.0


def test_abstention_leaves_the_wrong_candidate_untouched(sample_item):
    pair = get_axis("abstention").build(sample_item, random.Random(0), {})
    bw = dict(pair.base.blocks)
    pw = dict(pair.perturbed.blocks)
    wrong_label = "B" if pair.base.preferred == "A" else "A"
    assert bw[wrong_label] == pw[wrong_label]


def test_length_matched_control_reaches_cot_length(sample_item):
    cot = get_axis("verbose_cot").build(sample_item, random.Random(0), {})
    ctrl = get_axis("length_matched_control").build(sample_item, random.Random(0), {})
    cot_wrong = dict(cot.perturbed.blocks)["B" if cot.perturbed.preferred == "A" else "A"]
    ctrl_wrong = dict(ctrl.perturbed.blocks)["B" if ctrl.perturbed.preferred == "A" else "A"]
    assert len(ctrl_wrong) >= len(cot_wrong) * 0.9


def test_base_order_is_balanced_across_items(items):
    """Roughly half the items should put the correct answer in slot A."""
    a_slots = sum(1 for it in items if base_order(it)[0] == "A")
    assert 0.25 <= a_slots / len(items) <= 0.75


# --------------------------------------------------------------------- data


def test_item_pool_loads(items):
    assert len(items) >= 150


def test_item_pool_has_distinct_answers(items):
    for it in items:
        assert it.correct.text.strip() != it.wrong.text.strip()
        assert it.correct.is_correct and not it.wrong.is_correct


def test_item_pool_covers_all_domains(items):
    assert len({it.domain for it in items}) >= 4


# ------------------------------------------------------- pool-level confounds
#
# These guard the failure that invalidated the first pool: correct answers were
# systematically longer than wrong ones, so a judge's length preference read as
# competence and every axis measurement was contaminated. They are asserted
# here, not just in a script, because a pool that drifts back into imbalance
# must break the build rather than quietly produce publishable-looking garbage.


def _rel_gap(it) -> float:
    c, w = len(it.correct.text.strip()), len(it.wrong.text.strip())
    return (c - w) / max(c, w, 1)


def test_pool_is_not_length_confounded(items):
    """Mean signed length gap must be ~0, or length bias masquerades as skill."""
    gaps = [_rel_gap(it) for it in items]
    assert abs(sum(gaps) / len(gaps)) < 0.05, (
        f"mean signed length gap {sum(gaps) / len(gaps):+.3f}; "
        "the pool systematically favours one candidate on length"
    )


def test_no_item_is_grossly_length_imbalanced(items):
    """A few extreme items can carry the whole effect; cap them individually."""
    offenders = [(it.id, _rel_gap(it)) for it in items if abs(_rel_gap(it)) > 0.15]
    assert len(offenders) / len(items) <= 0.10, f"too many length-imbalanced items: {offenders[:10]}"


def test_pool_length_gap_is_not_directionally_skewed(items):
    """Balance in the mean can still hide a consistent lean on most items."""
    pos = sum(1 for it in items if _rel_gap(it) > 0.02)
    neg = sum(1 for it in items if _rel_gap(it) < -0.02)
    share = max(pos, neg) / len(items)
    assert share <= 0.65, (
        f"{share:.0%} of items lean the same way on length (pos={pos} neg={neg}); "
        "the pool leaks a length cue even if the mean looks fine"
    )


# ------------------------------------------------------------------ prompts


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("VERDICT: A\nREASON: x", Verdict.A),
        ("VERDICT: B\nREASON: x", Verdict.B),
        ("**VERDICT: A**", Verdict.A),
        ("verdict: b", Verdict.B),
        ("I refuse to compare these.", Verdict.PARSE_FAIL),
        ("", Verdict.PARSE_FAIL),
    ],
)
def test_parse_verdict(raw, expected):
    assert parse_verdict(raw) == expected


def test_prompt_contains_both_candidates(sample_item):
    pair = get_axis("length").build(sample_item, random.Random(0), {})
    prompt = build_prompt(sample_item.question, pair.perturbed)
    assert "CANDIDATE A" in prompt
    assert "CANDIDATE B" in prompt
    assert sample_item.question in prompt


def test_substance_template_warns_about_presentation(sample_item):
    pair = get_axis("length").build(sample_item, random.Random(0), {})
    assert "Ignore length" in build_prompt(sample_item.question, pair.perturbed, "v3_substance")
