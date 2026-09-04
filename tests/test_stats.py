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


def test_summarise_axis_no_discordance_is_not_significant():
    cells = [
        PairedOutcome(n=10, b=0, c=0, item_id="a"),
        PairedOutcome(n=10, b=0, c=0, item_id="b"),
    ]
    res = summarise_axis("length", "hyp", cells)
    assert res.p_mcnemar == 1.0
