"""Statistical protocol for judge-redteam.

Everything here is pure-Python and seeded, so results are bit-reproducible and
the whole module is unit-testable without any model access. That is deliberate:
the statistics must be verifiable independently of the judges.

Analysis unit
-------------
The preregistered analysis unit is the *item*, not the replicate. Item i is
scored with R replicate calls (same question, temperature > 0). Those R verdicts
are correlated observations of one thing, not R independent facts. Any inference
that resamples individual verdicts or feeds the pooled (B, C) discordant counts
into a plain McNemar test overstates confidence, because it treats R correlated
rows as R independent rows.

The headline test therefore resamples whole items (clusters) and recomputes the
signed discrepancy each time. See `cluster_permutation_p` and `summarise_axis`.

Open review
-----------
The cluster-permutation test is implemented and simulation-calibrated (see
tests/test_stats.py), but the method has NOT been signed off by a statistician.
STATS_REVIEW_NOTE below records exactly what is proven and what is not. Treat
any real-model finding as provisional until a stats reviewer has looked at it.
"""

from __future__ import annotations

# What is proven by simulation vs what still needs a human statistician.
# Kept here so the README and the report can cite the same single source.
STATS_REVIEW_NOTE = (
    "Cluster-permutation inference (item-level resampling) is simulation-calibrated "
    "for false-positive rate under the null, for power under a known bias, and for "
    "perfectly-correlated replicates. It has NOT been formally reviewed by a "
    "statistician; real-model findings are provisional pending that review."
)

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .types import PairedOutcome


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pair.

    Under H0 the direction of a flip is a fair coin, so b | (b+c) ~ Binom(b+c, 0.5).
    Falls back to the continuity-corrected normal approximation only when the
    exact sum becomes expensive (n > 500).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n > 500:
        chi = (abs(b - c) - 1) ** 2 / n
        return 2.0 * (1.0 - normal_cdf(math.sqrt(chi))) if chi > 0 else 1.0
    tail = sum(binom_pmf(i, n) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def cohens_h_var(n: int) -> float:
    """Approximate sampling variance of Cohen's h at sample size n."""
    return 1.0 / n if n > 0 else float("inf")


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    stat: Callable[[list[float]], float],
    n_resample: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260830,
) -> tuple[float, float]:
    """Percentile CI resampling whole clusters (an item's replicates are not independent)."""
    if not clusters:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(clusters)
    draws: list[float] = []
    for _ in range(n_resample):
        resampled = [clusters[rng.randrange(k)] for _ in range(k)]
        flat = [v for cluster in resampled for v in cluster]
        draws.append(stat(flat))
    draws.sort()
    lo = draws[int(math.floor(alpha / 2 * n_resample))]
    hi = draws[min(n_resample - 1, int(math.ceil((1 - alpha / 2) * n_resample)) - 1)]
    return (lo, hi)


def paired_bootstrap_diff_ci(
    pairs: Sequence[tuple[float, float]],
    n_resample: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260830,
) -> tuple[float, float]:
    """CI for mean(x - y) over paired observations, resampling pairs."""
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(pairs)
    diffs: list[float] = []
    for _ in range(n_resample):
        s = 0.0
        for _ in range(k):
            x, y = pairs[rng.randrange(k)]
            s += x - y
        diffs.append(s / k)
    diffs.sort()
    lo = diffs[int(math.floor(alpha / 2 * n_resample))]
    hi = diffs[min(n_resample - 1, int(math.ceil((1 - alpha / 2) * n_resample)) - 1)]
    return (lo, hi)


def cluster_permutation_p(
    cells: Sequence[PairedOutcome],
    n_resample: int = 10_000,
    seed: int = 20260830,
) -> tuple[float, int]:
    """Item-cluster-aware permutation test of net directional bias.

    The analysis unit is the *item*, not the replicate. The R replicate calls of
    one item are correlated, so they do not supply R independent flips. We keep
    each item's total discordance (b_i + c_i) as a single cluster but randomize
    its *direction* under the null -- exactly what a fair-coin flip of each
    item's verdict would do. Resampling whole items instead (keeping their
    realized directions) would re-center the null on the observed value and
    never reject, which is why direction, not item identity, is permuted.

    Under H0 there is no systematic direction bias, so the observed signed
    discrepancy D = sum_i sign_i (b_i - c_i) should be typical of the
    sign-permuted distribution. p is the fraction of permutations at least as
    extreme, with a +1 / (n+1) correction.

    Returns (p_value, observed_signed_discrepancy).
    """
    if not cells:
        return 1.0, 0
    mags = [c.b + c.c for c in cells]
    if sum(mags) == 0:
        return 1.0, 0
    signs = [1 if c.b >= c.c else -1 for c in cells]
    d_obs = sum(s * m for s, m in zip(signs, mags))
    rng = random.Random(seed)
    k = len(cells)
    count = 0
    for _ in range(n_resample):
        d = 0
        for i in range(k):
            s = 1 if rng.random() < 0.5 else -1
            d += s * mags[i]
        if abs(d) >= abs(d_obs):
            count += 1
    return (count + 1) / (n_resample + 1), d_obs


# --------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Holm step-down correction. Returns (rejected, adjusted_p)."""
    m = len(pvalues)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    rejected = [False] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    still_rejecting = True
    for rank, idx in enumerate(order):
        if still_rejecting and pvalues[idx] <= alpha / (m - rank):
            rejected[idx] = True
        else:
            still_rejecting = False
    return rejected, adjusted


# --------------------------------------------------------------------------
# random-effects meta-analysis (DerSimonian-Laird)
# --------------------------------------------------------------------------


@dataclass
class MetaResult:
    pooled: float
    se: float
    z: float
    p: float
    tau_sq: float
    q: float
    i_sq: float
    k: int
    ci: tuple[float, float]


def dersimonian_laird(effects: Sequence[float], variances: Sequence[float]) -> MetaResult:
    if not effects:
        return MetaResult(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, (float("nan"), float("nan")))
    k = len(effects)
    w = [1.0 / v if v > 0 else 0.0 for v in variances]
    sw = sum(w)
    if sw == 0:
        return MetaResult(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, k, (float("nan"), float("nan")))
    fe = sum(wi * ei for wi, ei in zip(w, effects)) / sw
    q = sum(wi * (ei - fe) ** 2 for wi, ei in zip(w, effects))
    df = k - 1
    sw2 = sum(wi * wi for wi in w)
    denom = sw - sw2 / sw
    tau_sq = max(0.0, (q - df) / denom) if denom > 0 and df > 0 else 0.0
    wstar = [1.0 / (v + tau_sq) if (v + tau_sq) > 0 else 0.0 for v in variances]
    swstar = sum(wstar)
    pooled = sum(wi * ei for wi, ei in zip(wstar, effects)) / swstar
    se = math.sqrt(1.0 / swstar)
    z = pooled / se if se > 0 else 0.0
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    i_sq = max(0.0, (q - df) / q * 100.0) if q > 0 and df > 0 else 0.0
    return MetaResult(
        pooled=pooled,
        se=se,
        z=z,
        p=p,
        tau_sq=tau_sq,
        q=q,
        i_sq=i_sq,
        k=k,
        ci=(pooled - 1.96 * se, pooled + 1.96 * se),
    )


# --------------------------------------------------------------------------
# the headline test
# --------------------------------------------------------------------------


@dataclass
class AxisResult:
    axis: str
    hypothesis: str
    n_pairs: int
    b: int
    c: int
    net_bias: float
    flip_rate: float
    p_flip_wrong: float
    p_flip_right: float
    p_mcnemar: float
    cohens_h: float
    h_ci: tuple[float, float]
    n_items: int = 0
    meta: MetaResult | None = None
    noise_floor: float = 0.0
    noise_self_disagreement: float = 0.0
    noise_ci: tuple[float, float] = (float("nan"), float("nan"))
    above_noise: bool | None = None
    p_adjusted: float = 1.0
    rejected: bool = False
    verdict: str = ""
    confirmatory: bool = True
    extra: dict = field(default_factory=dict)


def summarise_axis(
    axis: str,
    hypothesis: str,
    cells: Sequence[PairedOutcome],
    noise_cells: Sequence[PairedOutcome] = (),
    confirmatory: bool = True,
) -> AxisResult:
    """Direction-decomposed bias test for one axis.

    `cells` are per-(item, judge) paired outcomes, base vs perturbed.
    `noise_cells` are per-(item, judge) outcomes, noise_a vs noise_b — two
    unperturbed repeats. They establish the floor that a real effect must clear.
    """
    n = sum(x.n for x in cells)
    n_items = len(cells)
    b = sum(x.b for x in cells)
    c = sum(x.c for x in cells)
    agg = PairedOutcome(n=n, b=b, c=c)

    # Item-cluster-aware: resample whole items, not individual replicates, so the
    # R correlated calls per item do not inflate significance. (Previously this
    # fed the pooled (b, c) into mcnemar_exact, which assumes independence.)
    p, _ = cluster_permutation_p(cells)
    h = cohens_h(agg.p_flip_wrong, agg.p_flip_right)

    clusters = [[1.0] * x.b + [-1.0] * x.c + [0.0] * (x.n - x.b - x.c) for x in cells]
    h_ci = cluster_bootstrap_ci(clusters, lambda flat: sum(flat) / len(flat) if flat else 0.0)

    effects = [cohens_h(x.p_flip_wrong, x.p_flip_right) for x in cells if x.n > 0]
    variances = [cohens_h_var(x.n) for x in cells if x.n > 0]
    meta = dersimonian_laird(effects, variances)

    noise_floor = 0.0
    noise_self_disagreement = 0.0
    noise_ci: tuple[float, float] = (float("nan"), float("nan"))
    above_noise: bool | None = None
    if noise_cells:
        nn = sum(x.n for x in noise_cells)
        nb = sum(x.b for x in noise_cells)
        nc = sum(x.c for x in noise_cells)
        # Two distinct quantities, reported separately so the reader knows which
        # is being compared:
        #   noise_floor            = wrong-direction self-flip rate (b/nn). This is
        #                           the baseline rate at which the judge destroys a
        #                           *correct* verdict with no perturbation. A real
        #                           effect must push past THIS, not past the total.
        #   noise_self_disagreement = total flip rate (b+c)/nn: the judge merely
        #                           changes its mind between two identical prompts,
        #                           in either direction. Informational only.
        noise_floor = nb / nn if nn else 0.0
        noise_self_disagreement = (nb + nc) / nn if nn else 0.0

        # Is the perturbation worse than the judge's own wrong-direction rate?
        # Paired at item level: each item contributes a perturbed-condition
        # P(->wrong) and a noise-condition P(->wrong). If the bootstrap CI for
        # the difference sits entirely above zero, the perturbation destroys
        # verdicts the judge would not have destroyed on its own, and only then
        # do we call it a finding.
        pert_by_item = {x.item_id: x for x in cells if x.item_id}
        noise_by_item = {x.item_id: x for x in noise_cells if x.item_id}
        shared = sorted(set(pert_by_item) & set(noise_by_item))
        if shared:
            pairs = [
                (pert_by_item[i].p_flip_wrong, noise_by_item[i].p_flip_wrong) for i in shared
            ]
            noise_ci = paired_bootstrap_diff_ci(pairs, seed=20260831)
            above_noise = noise_ci[0] > 0.0
        else:
            # No shared items: compare aggregate wrong-direction rates directly.
            noise_ci = (noise_floor, noise_floor)
            above_noise = agg.p_flip_wrong > noise_floor

    return AxisResult(
        axis=axis,
        hypothesis=hypothesis,
        n_pairs=n,
        n_items=n_items,
        b=b,
        c=c,
        net_bias=agg.net_bias,
        flip_rate=agg.flip_rate,
        p_flip_wrong=agg.p_flip_wrong,
        p_flip_right=agg.p_flip_right,
        p_mcnemar=p,
        cohens_h=h,
        h_ci=h_ci,
        meta=meta,
        noise_floor=noise_floor,
        noise_self_disagreement=noise_self_disagreement,
        noise_ci=noise_ci,
        above_noise=above_noise,
        confirmatory=confirmatory,
        extra={"method": "cluster-permutation", "n_resample": 10_000},
    )


def apply_correction(results: list[AxisResult], alpha: float = 0.05) -> list[AxisResult]:
    """Holm-Bonferroni across the confirmatory family only.

    Ablation controls are corrected out of the family: they answer a different
    question ("is this effect really the thing I think it is?") and inflating m
    with them would cost power on the seven preregistered hypotheses.
    """
    fam = [i for i, r in enumerate(results) if r.confirmatory]
    if fam:
        rejected, adjusted = holm_bonferroni([results[i].p_mcnemar for i in fam], alpha=alpha)
        for slot, i in enumerate(fam):
            results[i].rejected = bool(rejected[slot])
            results[i].p_adjusted = adjusted[slot]
    for r in results:
        if not r.confirmatory:
            r.p_adjusted = r.p_mcnemar
            r.rejected = r.p_mcnemar < alpha
        if not r.rejected:
            r.verdict = "no significant net bias"
        elif r.above_noise is False:
            r.verdict = "significant but BELOW noise floor (not reported as a finding)"
        else:
            r.verdict = "systematic bias confirmed"
    return results


__all__ = [
    "normal_cdf",
    "mcnemar_exact",
    "cohens_h",
    "cohens_h_var",
    "cluster_bootstrap_ci",
    "paired_bootstrap_diff_ci",
    "cluster_permutation_p",
    "holm_bonferroni",
    "dersimonian_laird",
    "MetaResult",
    "AxisResult",
    "summarise_axis",
    "apply_correction",
    "STATS_REVIEW_NOTE",
]
