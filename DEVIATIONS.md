# Deviations from preregistration

Empty as of 2026-08-30.

Every entry records:

- the date,
- what changed,
- why,
- and **whether the change was made before or after inspecting outcome data**.

That last field is the only one that matters for credibility. A change made
after seeing results is a hypothesis-generating exercise, not a test, and must
be labelled as such.

## Amendment 1 (2026-08-30) — recorded before any data collection

**Changed:** target item count raised from `N ≥ 120` to `N ≥ 150`; pilot size and
its minimum detectable effect documented explicitly.

**Why:** the a-priori power analysis mandated by section 5 was run and returned
measured values rather than the assumed ones. At a 15% noise floor, 120 items
gives 71% power at `h = 0.2`, short of the preregistered 80%. The target was
raised to the size that actually achieves it (~150). The 40-item seed pool is
reclassified as a pilot, and section 5 now states that a null on the pilot means
"not detected", not "absent".

**Timing:** before outcome data existed. No judge had been run.

## Amendment 2 (2026-08-30) — recorded before any data collection

**Changed:** H1 (position) pins the correct answer to slot A in the base
condition instead of using the per-item randomised slot assignment used by every
other axis. Added the `length_matched_control` ablation axis for H5.

**Why:** discovered while calibrating the instrument against a simulated judge
with a known position preference. The randomised assignment cancelled the effect
by construction — the bias helped the base condition on half the items and hurt
it on the other, collapsing `net_bias` to zero for a judge that demonstrably had
a strong slot preference. Randomisation is correct for the other six axes and
actively wrong for this one. Separately, the H5 perturbation is necessarily
longer than the baseline, so it confounds with H2; the length-matched control
separates the two.

**Timing:** before outcome data existed. Found during instrument calibration,
which is exactly what calibration is for.

## Amendment 3 (2026-08-30) — recorded before any real-judge data collection

**Changed:** the 40-item seed pool was **discarded and rebuilt**, not expanded.
The shipped pool is now 150 items (30 per domain) in `data/items.jsonl`, sourced
from `data/raw/<domain>.jsonl` and assembled by `scripts/build_pool.py`. Candidate
length balance was added to the preregistration as a binding acceptance criterion
(§4), with a new gate (`scripts/validate_items.py`) and four new invariant tests.

**Why:** an audit of the first pool found a confound severe enough to invalidate
any result produced with it. Correct answers were **longer than wrong ones on 32
of 40 items, by 42% on average** (mean signed length gap +0.242; 70% of items over
a 15% cap; correct answers carried more digits on 65% of items vs 8%).

This is not a precision problem. If the correct answer is systematically longer,
then a judge that prefers longer answers is rewarded for a *presentation
preference* as though it were *discrimination*. The length axis (H2) becomes
unmeasurable, and every other axis inherits the same contamination through its
base condition. The failure mode is a confident, reproducible, wrong answer — the
one kind that publishes.

The rebuild enforces balance by construction: each item uses an identical
reasoning skeleton with one step divergent, so the two candidates diverge on
correctness and not on surface properties. Measured on the new pool: mean signed
gap +0.003, maximum item gap 12%, zero items over cap.

The old file was deleted rather than kept as a pilot. A pool with a known
fatal confound is a trap for whoever reaches for it next.

**Timing:** before any real judge was run. The defect was found by auditing the
inputs, and the simulated-judge runs are instrument calibration, not outcome
data — they measure the harness, not any model.

## Amendment 4 (2026-08-30) — recorded before any real-judge data collection

**Changed:** the noise-floor criterion was rewritten from a point-estimate
threshold (`p_flip_wrong > noise_ci_upper + noise_floor * 0.5`) to a
*between-cluster* two-proportion test: perturbation flip rate versus
self-disagreement flip rate, clustered by item.

**Why:** the original form was a magic number with no statistical justification,
sitting at the exact centre of a method whose entire claim is separating bias from
variance. An ad-hoc constant there undermines every verdict downstream of it.

**Timing:** before outcome data existed.

## Template

```
## Amendment N (YYYY-MM-DD) — recorded before/after outcome data

**Changed:** ...
**Why:** ...
**Timing:** ...
```
