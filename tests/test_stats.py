"""Statistical correctness tests for judge-redteam.

These are pure-Python simulations: no model, no network. They exist to prove
that the inference is honest about its sample size. The analysis unit is the
*item*; the R replicates of one item are correlated, not independent. A test
that ignores that overstates confidence.

All simulations are seeded, so the numbers are deterministic and the bounds
below are regression guards, not loosened flakiness.
"""

from __future__ import annotations

import random

import pytest

from jrt.stats import cluster_permutation_p, mcnemar_exact, summarise_axis
from jrt.types import PairedOutcome

ALPHA = 0.05


def _sim_cells(rng, *, n_items=24, reps=(3, 12), bias=0.0, active_frac=0.5):
    """Synthetic per-item paired outcomes.

    Each item may be 'active' (it produces discordant flips) or silent. Active
    items flip every one of their replicates in a single direction, so the
    replicates are perfectly correlated -- the realistic cluster structure. Under
    H0 (bias=0) the direction is a fair coin; with bias>0 the judge leans wrong.
    """
    cells = []
    for i in range(n_items):
        n_i = rng.randint(*reps)
        if rng.random() < active_frac:
            s = 1 if rng.random() < (0.5 + bias) else -1
            b_i, c_i = (n_i, 0) if s > 0 else (0, n_i)
        else:
            b_i = c_i = 0
        cells.append(PairedOutcome(n=n_i, b=b_i, c=c_i, item_id=f"i{i}"))
    return cells


def _false_positive_rate(cluster=True, **kw):
    rej = 0
    total = 0
    for t in range(120):
        rng = random.Random(1000 + t)
        cells = _sim_cells(rng, **kw)
        if cluster:
            p, _ = cluster_permutation_p(cells, n_resample=1000, seed=2000 + t)
        else:
            B = sum(c.b for c in cells)
            C = sum(c.c for c in cells)
            p = mcnemar_exact(B, C)
        total += 1
        if p < ALPHA:
            rej += 1
    return rej / total


def test_cluster_fpr_calibrated_under_null():
    # Under no real bias, the cluster test must reject at roughly alpha, not more.
    fpr = _false_positive_rate(cluster=True, bias=0.0)
    assert 0.0 <= fpr <= 0.12, f"cluster FPR {fpr:.3f} should be near {ALPHA}"


def test_cluster_power_under_known_bias():
    # With a real per-item lean toward wrong flips, the test must usually fire.
    rej = 0
    total = 0
    for t in range(120):
        rng = random.Random(5000 + t)
        cells = _sim_cells(rng, bias=0.4)
        p, _ = cluster_permutation_p(cells, n_resample=1000, seed=6000 + t)
        total += 1
        if p < ALPHA:
            rej += 1
    power = rej / total
    assert power >= 0.75, f"cluster power {power:.3f} too low"


def test_naive_pooled_inflates_fpr_vs_cluster():
    # The bug this hardening fixes: treating R correlated replicates as R
    # independent pairs. Under item-level chunking the naive exact McNemar
    # rejects far more often than it should; the cluster test stays calibrated.
    cluster_fpr = _false_positive_rate(cluster=True, bias=0.0)
    naive_fpr = _false_positive_rate(cluster=False, bias=0.0)
    assert naive_fpr >= cluster_fpr + 0.03, (
        f"naive FPR {naive_fpr:.3f} should exceed cluster {cluster_fpr:.3f}"
    )
    assert naive_fpr >= 0.08, f"naive FPR {naive_fpr:.3f} should be inflated"


def test_summarise_axis_records_item_level_identity():
    # Seven items all leaning wrong under perturbation, plus one neutral item.
    # With enough items the sign-permutation null is tight enough to reject.
    cells = [PairedOutcome(n=10, b=8, c=2, item_id=f"i{j}") for j in range(7)]
    cells.append(PairedOutcome(n=10, b=5, c=5, item_id="neutral"))
    res = summarise_axis("position", "Perturbation reduces correct verdicts", cells)
    assert res.n_items == 8
    assert res.n_pairs == 80
    assert res.p_mcnemar < 0.05
    assert res.extra.get("method") == "cluster-permutation"


def test_balanced_flips_within_every_item_are_exactly_null():
    cells = [PairedOutcome(n=10, b=5, c=5, item_id=f"i{j}") for j in range(8)]
    p, discrepancy = cluster_permutation_p(cells)
    assert discrepancy == 0
    assert p == 1.0


def test_reported_interval_is_for_cohens_h():
    cells = [PairedOutcome(n=10, b=8, c=2, item_id=f"i{j}") for j in range(12)]
    res = summarise_axis("position", "hyp", cells)
    assert res.h_ci[0] <= res.cohens_h <= res.h_ci[1]


def test_summarise_axis_no_discordance_is_not_significant():
    cells = [
        PairedOutcome(n=10, b=0, c=0, item_id="a"),
        PairedOutcome(n=10, b=0, c=0, item_id="b"),
    ]
    res = summarise_axis("length", "hyp", cells)
    assert res.p_mcnemar == 1.0


def _paired(cells, noise, axis="position", hyp="Perturbation hurts", confirmatory=True):
    return summarise_axis(axis, hyp, cells, noise_cells=noise, confirmatory=confirmatory)


def test_noise_floor_is_wrong_direction_rate_not_total():
    # Each item: perturbation flips 50% toward wrong; noise flips 10% toward wrong
    # but 50% toward right. The reported noise floor must be the 10% wrong rate,
    # not the 60% total self-disagreement rate.
    cells = [PairedOutcome(n=20, b=10, c=2, item_id=f"i{j}") for j in range(5)]
    noise = [PairedOutcome(n=20, b=2, c=10, item_id=f"i{j}") for j in range(5)]
    res = _paired(cells, noise)
    assert res.noise_floor == 0.1
    assert res.noise_self_disagreement == pytest.approx(0.6)


def test_above_noise_true_when_perturbation_exceeds_floor():
    cells = [PairedOutcome(n=20, b=15, c=1, item_id=f"i{j}") for j in range(5)]
    noise = [PairedOutcome(n=20, b=2, c=10, item_id=f"i{j}") for j in range(5)]
    res = _paired(cells, noise)
    assert res.noise_floor == pytest.approx(0.1)
    assert res.above_noise is True


def test_above_noise_false_when_equal_to_floor():
    # Boundary: perturbation wrong-rate equals the noise wrong-rate -> not above.
    cells = [PairedOutcome(n=20, b=10, c=10, item_id=f"i{j}") for j in range(5)]
    noise = [PairedOutcome(n=20, b=10, c=10, item_id=f"i{j}") for j in range(5)]
    res = _paired(cells, noise)
    assert res.noise_floor == pytest.approx(0.5)
    assert res.above_noise is False


def test_above_noise_false_when_below_floor():
    cells = [PairedOutcome(n=20, b=2, c=10, item_id=f"i{j}") for j in range(5)]
    noise = [PairedOutcome(n=20, b=10, c=2, item_id=f"i{j}") for j in range(5)]
    res = _paired(cells, noise)
    assert res.above_noise is False


def test_significant_but_below_noise_is_not_confirmed():
    # Strong net (p significant) but the perturbation's wrong-rate is below the
    # judge's own spontaneous wrong-rate: it must NOT be reported as a finding.
    cells = [PairedOutcome(n=20, b=16, c=2, item_id=f"i{j}") for j in range(6)]
    noise = [PairedOutcome(n=20, b=20, c=0, item_id=f"i{j}") for j in range(6)]
    res = _paired(cells, noise)
    assert res.p_mcnemar < 0.05
    assert res.above_noise is False
    from jrt.stats import apply_correction

    out = apply_correction([res])
    assert "BELOW noise floor" in out[0].verdict
    assert out[0].verdict != "systematic bias confirmed"
