"""A-priori power analysis, preregistered in PREREGISTRATION.md section 5.

Two questions get answered:
  1. How many items are needed to detect a given effect at 80% power?
  2. With the items we actually have, what is the smallest effect we can detect?

Both are answered by Monte Carlo under a conservative item-cluster model:
replicates of an active item move together, so five calls on one question do
not count as five independent facts. The resulting exact sign test is the
equal-weight special case of the headline item-cluster permutation test.

Run:  python scripts/power_analysis.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jrt.stats import cohens_h, mcnemar_exact  # noqa: E402

ALPHA = 0.05
N_HYPOTHESES = 7
NOISE_FLOOR = 0.15


def delta_for_h(target_h: float, noise_floor: float) -> float:
    """Invert Cohen's h to find the net bias that produces it.

    The perturbation is modelled as tilting the judge's own flip rate rather
    than adding new flips: p_wrong + p_right = noise_floor stays fixed while
    p_wrong - p_right = delta grows. That is the conservative model — it assumes
    the bias only redirects verdicts the judge was already going to flip.
    """
    lo, hi = 0.0, noise_floor
    for _ in range(200):
        mid = (lo + hi) / 2
        pw = (noise_floor + mid) / 2
        pr = (noise_floor - mid) / 2
        if cohens_h(pw, pr) < target_h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def power_at(
    effect_h: float,
    n_items: int,
    reps: int,
    noise_floor: float = NOISE_FLOOR,
    alpha: float = ALPHA,
    m: int = N_HYPOTHESES,
    n_sims: int = 4000,
    seed: int = 7,
) -> float:
    delta = delta_for_h(effect_h, noise_floor)
    p_wrong = (noise_floor + delta) / 2
    p_right = (noise_floor - delta) / 2
    # Conservative cluster model: an item's replicates are perfectly correlated.
    # `reps` affects the precision of its observed rate, not the number of
    # independent experimental units, so the sign test has n_items trials.
    del reps
    trials = n_items
    threshold = alpha / m  # Holm is at least as powerful as Bonferroni
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_sims):
        # Draw the three mutually exclusive item outcomes directly. This works
        # on every supported Python version and cannot count one item in both
        # flip directions, as two independent binomial draws could.
        b = c = 0
        cut_right = p_wrong + p_right
        for _item in range(trials):
            u = rng.random()
            if u < p_wrong:
                b += 1
            elif u < cut_right:
                c += 1
        if mcnemar_exact(b, c) < threshold:
            hits += 1
    return hits / n_sims


def min_detectable_h(
    n_items: int,
    reps: int,
    target_power: float = 0.80,
    noise_floor: float = NOISE_FLOOR,
    **kwargs,
) -> float:
    lo, hi = 0.01, 1.2
    for _ in range(24):
        mid = (lo + hi) / 2
        if power_at(mid, n_items, reps, noise_floor, **kwargs) < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    print("A-priori power analysis")
    print(f"  alpha={ALPHA}, {N_HYPOTHESES} confirmatory hypotheses, Holm correction")
    print(f"  noise floor (judge self-disagreement) assumed = {NOISE_FLOOR:.0%}")
    print()

    print("1) Items required for 80% power at a given effect size")
    print("   (Cohen's h: 0.2 small, 0.5 medium, 0.8 large)")
    print()
    print("   | h | items needed |")
    print("   |---|---:|")
    for h in (0.2, 0.3, 0.5, 0.8):
        needed = None
        for n in range(20, 700, 10):
            if power_at(h, n, 5) >= 0.80:
                needed = n
                break
        print(f"   | {h:.1f} | {needed if needed else '>690'} |")

    print()
    print("2) Smallest detectable effect with the current item pool, at 5 reps")
    print()
    print("   | items | min detectable h |")
    print("   |---|---:|")
    for n in (40, 80, 120, 200, 300):
        print(f"   | {n} | {min_detectable_h(n, 5):.3f} |")

    print()
    print("3) Power at h=0.2 for candidate pool sizes")
    print()
    print("   | items | power |")
    print("   |---|---:|")
    for n in (40, 80, 120, 200, 300):
        print(f"   | {n} | {power_at(0.2, n, 5):.1%} |")

    current = 150
    mde = min_detectable_h(current, 5)
    print()
    print(
        f"Verdict: {current} items detect h >= {mde:.3f} at 80% power. "
        "Small effects will be missed, and a null result on a small pool means "
        "'not detected', never 'absent'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
