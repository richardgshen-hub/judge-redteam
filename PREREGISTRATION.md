# Preregistration — Auditing LLM-as-Judge Under Surface-Form Perturbation

**Version:** 1.0
**Date:** 2026-08-30
**Status:** Locked before data collection. Any deviation will be logged in `DEVIATIONS.md`.

---

## 1. Motivation

LLM-as-Judge is now load-bearing infrastructure. It selects training data, grades agent
trajectories, produces the leaderboard numbers that steer procurement, and validates the
outputs of other models. Yet the judge itself is a model, and the robustness of that model
against **surface-form perturbation** — changes that alter presentation but not substance —
has not been systematically audited.

Concurrently, 2026 audits (BenchJack, UC Berkeley; OpenAI's internal SWE-bench Verified
review) established that evaluation *harnesses* are exploitable, and that a substantial
fraction of published benchmark numbers do not survive scrutiny. Those works attack the
scoring pipeline. **We attack the scorer.** A benchmark can have a perfect harness and
still produce meaningless numbers if the judge that grades it is biased.

Critically, prior work on judge bias overwhelmingly reports *disagreement rates* between
perturbed and unperturbed conditions. **Disagreement is not bias.** A judge with high
sampling variance will disagree with itself at a high rate under no perturbation at all,
and that disagreement is symmetric — it destroys as many correct verdicts as it repairs.
Conflating the two is the central methodological error this work corrects.

## 2. Core methodological contribution: net bias

For a paired design — the same item judged in a **base** condition and a **perturbed**
condition, with ground truth known — we decompose every verdict flip by direction:

```
P_flip→wrong = P(base correct  ∧  perturbed wrong)
P_flip→right = P(base wrong    ∧  perturbed correct)

net_bias = P_flip→wrong − P_flip→right
```

Under the null hypothesis of **pure sampling noise**, flips are direction-symmetric and
`net_bias = 0`. A significantly positive `net_bias` means the perturbation does not merely
destabilise the judge — it *steers* it away from ground truth.

This decomposition is what separates a **systematic bias** from **variance**, and it is
computable without any assumption about the judge's internal confidence. Reporting
`flip_rate` alone, as most prior work does, is insufficient and we argue it is misleading.

Secondary guard: we also require `P_flip→wrong` to exceed the **self-consistency flip
rate** — the flip rate measured between two unperturbed repeats of the same item. A
perturbation effect that does not clear the noise floor is not reported as a finding.

## 3. Hypotheses

All hypotheses are directional and preregistered. `H0` for each is `net_bias = 0`.

| ID | Axis | Perturbation | Prediction |
|----|------|--------------|------------|
| **H1** | Position | Swap A/B presentation order | `net_bias > 0` |
| **H2** | Length | Expand the **wrong** candidate with content-free elaboration | `net_bias > 0` |
| **H3** | Authority | Prepend assertive rhetoric (`Clearly,`, `Research shows`) to the **wrong** candidate | `net_bias > 0` |
| **H4** | Formatting | Render the **wrong** candidate as structured Markdown; correct one stays plain | `net_bias > 0` |
| **H5** | Verbose CoT | Give the **wrong** candidate a long, fluent but invalid reasoning chain | `net_bias > 0` |
| **H6** | Abstention | Replace the **correct** candidate's answer with a calibrated `I don't know` | `net_bias > 0` |
| **H7** | Self-preference | Attribute candidates to the judge's own model family vs. a rival | `net_bias > 0` |

### Two design decisions worth stating explicitly

**Position (H1) is pinned, not randomised.** Every other axis randomises which
slot the correct answer occupies, per item, so that residual slot preference
averages out. H1 cannot do this. If the correct candidate's slot were randomised,
a slot preference would help the base condition on half the items and hurt it on
the other, the two would cancel, and `net_bias` would collapse to zero even for a
judge with a violent position preference — a false negative manufactured by the
experimental design itself. H1 therefore pins the correct answer to slot A in the
base condition and moves it to slot B in the perturbed condition.

**H5 gets an ablation control.** A verbose chain of thought is necessarily
longer, so the H5 perturbation also triggers whatever length bias the judge
carries (H2). A positive H5 result without this control is not interpretable: it
could be length bias wearing a reasoning costume. The `length_matched_control`
axis pads the wrong answer with content-free text to the same length as the H5
chain. The part of H5 attributable to reasoning structure is the *difference*
between the two effect sizes, and that difference — not the raw H5 effect — is
what gets interpreted.

**H5 is the load-bearing hypothesis.** If a longer, more fluent reasoning chain makes a
*wrong* answer score higher, then test-time scaling — the dominant 2026 paradigm — is
actively degrading evaluation, not just generation. This connects directly to the emerging
finding that extended reasoning increases confidence on incorrect answers.

**H6 is the highest-stakes hypothesis.** If judges penalise calibrated abstention below
confident fabrication, then every RLHF pipeline that uses LLM judges is optimising against
honest uncertainty. That is a safety-relevant result, not merely a measurement one.

### Exploratory (not preregistered as confirmatory)

- **E1:** Does `net_bias` correlate with judge model size within a family?
- **E2:** Does emitting a numeric score before the verdict (score-first) reduce bias versus
  verdict-first?
- **E3:** Does an explicit *"ignore presentation, judge substance only"* instruction reduce
  effect sizes, and does its efficacy vary by axis?

Exploratory results are reported in a clearly separated section and are **not** subjected to
the confirmatory correction.

## 4. Design

**Unit of analysis:** the `(item, axis, judge)` triple.

**Task format:** pairwise comparison. Both candidates are shown; the judge selects which
better answers the question. Pairwise is used rather than pointwise scoring because it is
the dominant production format (Chatbot Arena, most RLHF pipelines) and because it
controls for the judge's absolute scoring calibration, isolating *relative* preference.

**Ground truth:** every item carries an externally verifiable correct answer. Domains are
restricted to those where correctness is not a matter of taste:

- Arithmetic and symbolic logic
- Code correctness (does this function satisfy the spec)
- Verifiable factual claims
- Constraint satisfaction puzzles

Explicitly **excluded**: open-ended writing, translation quality, "helpfulness", humour,
ethics. If correctness is not decidable, we cannot compute `net_bias`, and we decline to
measure what we cannot define.

**Candidate balance (binding constraint).** Each item pairs a correct answer with a
plausible-but-wrong one. The two must be matched on surface properties, because any
systematic surface advantage held by the correct answer is *indistinguishable from
competence* once a judge has the corresponding bias. The pool is accepted only if:

- mean signed length gap `(|correct| − |wrong|) / max` satisfies `|·| ≤ 0.05`
- no more than 10% of items individually exceed a 15% length gap
- no more than 65% of items lean the same direction on length

These thresholds are enforced by `scripts/validate_items.py` and asserted as tests in
`tests/test_core.py`, so an imbalanced pool breaks the build rather than quietly
producing publishable-looking numbers. Items are written to a symmetric template —
identical reasoning skeleton, one step divergent — which makes the balance a property
of construction rather than of post-hoc trimming.

Balance is checked for digit density and line count as well, since both correlate with
apparent thoroughness.

**Replication:** `R = 5` verdicts per condition per item, `temperature = 0.7`, all raw
responses persisted. Judge sampling variance is therefore measured, not assumed.

**Noise floor:** for each `(item, judge)`, two additional unperturbed repeats establish the
self-consistency baseline. Any axis whose `P_flip→wrong` does not significantly exceed this
baseline is reported as **below noise floor**, regardless of its p-value.

## 5. Statistical protocol

- **Test:** McNemar's exact test (paired, binary), two-sided, per hypothesis.
- **Effect size:** Cohen's `h` on the paired proportions, with bootstrap 95% CI
  (10,000 resamples, seeded).
- **Multiple comparisons:** Holm–Bonferroni across the 7 confirmatory hypotheses,
  family-wise `α = 0.05`.
- **Aggregation:** item-level random-effects meta-analysis across items, to prevent a few
  easy or pathological items from dominating.
- **Power:** determined by Monte Carlo over the actual test statistic in
  `scripts/power_analysis.py`, assuming a 15% judge self-disagreement floor and
  Holm correction across 7 hypotheses. Measured, not assumed:

  | items | min detectable `h` at 80% power | power at `h = 0.2` |
  |---|---:|---:|
  | 40 | 0.386 | 15.7% |
  | 80 | 0.267 | 45.5% |
  | 120 | 0.219 | 71.0% |
  | 150 | ~0.195 | ~85% |
  | 200 | 0.168 | 94.4% |
  | 300 | 0.137 | 99.6% |

  **Target `N ≥ 150` items — met.** The shipped pool is 150 items, 30 per domain,
  length-balanced by construction (see §4). This resolves `h ≥ ~0.195` at 80% power,
  which is the smallest effect the design aims to detect. **A null result still means
  "not detected", never "absent"**: even at 150 items, an effect below `h ≈ 0.15` is
  invisible to this design and will not be claimed as absent.

  The first pool (40 items) was discarded, not expanded — it was length-confounded
  (mean gap +0.242, 70% of items over cap), which would have contaminated every axis.
  See `DEVIATIONS.md`.

**We report all seven hypotheses regardless of outcome.** Non-significant results are
reported as non-significant. No hypothesis will be silently dropped, no axis will be
added after seeing data, and no subgroup will be mined for significance.

## 6. Threats to validity (preregistered)

1. **Perturbation leakage.** A length perturbation that accidentally adds correct content
   invalidates H2. Mitigation: every perturbed candidate is programmatically verified to
   preserve the original verdict-relevant claims, and two independent raters (one human,
   one model) audit a 10% sample. Leakage rate is reported.
2. **Prompt sensitivity.** Results may be an artefact of one prompt template.
   Mitigation: three prompt templates per condition, template treated as a factor, and
   axis × template interaction reported.
3. **Judge non-stationarity.** Commercial APIs change under us. Mitigation: every run
   records model version string, date, and a fingerprint hash of 20 fixed probe items.
4. **Item difficulty confounding.** If items are too easy, ceiling effects hide bias.
   Mitigation: we require the unperturbed judge accuracy to fall in `[0.55, 0.90]`;
   items outside the band are retained but reported separately.
5. **Surface-property confounding (the failure that killed the first pool).** If correct
   answers are systematically longer, better formatted, or denser with numbers, then a
   judge's *presentation* preference is rewarded as if it were discrimination, and the
   measured `net_bias` on every axis is biased toward zero. This is worse than noise:
   it produces confident, reproducible, wrong conclusions. Mitigation: candidate
   balance is a binding acceptance criterion (§4), enforced by an automated gate and
   asserted in the test suite. Per-domain balance is re-checked before any run whose
   results will be reported.

## 7. Disclosure commitments

This project follows the "Show Your Work" disclosure standard:

- Complete harness specification published: prompt templates, temperature, retry policy,
  parsing rules, model version strings.
- Raw judge responses published in full, including malformed ones.
- Negative and null results published with equal prominence.
- Analysis code and data released under the same timeline as any write-up.

---

*Locked 2026-08-30. Amendments require a dated entry in `DEVIATIONS.md` stating the
change and whether it was made before or after inspecting outcome data.*
