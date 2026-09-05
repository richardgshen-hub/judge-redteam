# judge-redteam

[![CI](https://github.com/richardgshen-hub/judge-redteam/actions/workflows/ci.yml/badge.svg)](https://github.com/richardgshen-hub/judge-redteam/actions/workflows/ci.yml)

LLM-as-judge is everywhere — RLHF pipelines, agent benchmarks, data curation — and nobody audits the auditor. This repo is a preregistered harness that applies controlled surface-form, metadata, and behavioral interventions to answers, then measures whether an LLM judge is steered away from ground truth. **Status: the instrument is built, calibrated on simulated judges, and unit-tested (90 tests + CI); real-model experiments have not been run yet, so this repo contains no conclusions about any real model.**

> **Read this first:** every number currently in this repository comes from a
> *simulated* judge with deliberately injectable bias. They demonstrate that the
> measurement pipeline works. They are **not** findings about GPT, Claude,
> Qwen, or any other model.

---

## How it works

```
                 ┌──────────────────────────────────────────────┐
                 │ 150-item balanced pool (5 domains)           │
                 │ length/digit-balanced by construction        │
                 └──────────────────┬───────────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ for each item × axis: build a PAIRED presentation     │
        │   base        = neutral presentation                  │
        │   perturbed   = presentation + one perturbation       │
        │   noise_a/b   = two identical unperturbed repeats     │
        └──────────────────────────┬────────────────────────────┘
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ judge scores every presentation, R replicates each,   │
        │ every verdict persisted + fsynced (crash-safe resume) │
        └──────────────────────────┬────────────────────────────┘
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ direction decomposition per item:                     │
        │   net_bias = P(correct→wrong) − P(wrong→correct)      │
        │ item-cluster permutation test + noise-floor gate      │
        └──────────────────────────┬────────────────────────────┘
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ report: effect sizes, CIs, Holm-adjusted p,           │
        │ effective sample size, exclusions, disclosure card    │
        └───────────────────────────────────────────────────────┘
```

Two design decisions carry most of the weight:

**1. Direction, not disagreement.** A noisy judge disagrees with itself constantly under no perturbation at all, symmetrically — so raw disagreement rates are meaningless. Every flip is decomposed:

```
net_bias = P(base correct ∧ perturbed wrong) − P(base wrong ∧ perturbed correct)
```

Pure noise gives `net_bias = 0`. A positive `net_bias` that clears the noise floor means the perturbation is *steering* the judge, not merely destabilising it. The noise floor itself is the judge's own **wrong-direction self-flip rate** between two identical unperturbed repeats — and an axis below that floor is reported as *not detected*, never as a finding, even if p < α.

**2. The item is the analysis unit.** R replicate calls on the same question are correlated observations, not R independent facts. The significance test keeps each item's observed net directional discrepancy (`b-c`) together and randomly flips its sign (`stats.cluster_permutation_p`), so a judge tested with 5 replicates does not get 5× the statistical confidence. This method is simulation-calibrated (false-positive rate, power, correlated replicates) but **not yet reviewed by a statistician** — see `stats.STATS_REVIEW_NOTE`.

---

## The seven hypotheses

Preregistered in [PREREGISTRATION.md](PREREGISTRATION.md) before any data collection. Original numbering retained; v0.2 reclassified two of them to match what they actually manipulate (logged in [DEVIATIONS.md](DEVIATIONS.md) amendments 5–6).

| ID | Axis | Category | Manipulation |
|----|------|----------|--------------|
| H1 | `position` | surface-form | Swap the A/B presentation order |
| H2 | `length` | surface-form | Pad the **wrong** answer with content-free elaboration |
| H3 | `authority` | surface-form | Prepend assertive rhetoric to the **wrong** answer |
| H4 | `format` | surface-form | Render the **wrong** answer as structured Markdown |
| H5 | `verbose_cot` | surface-form | Give the **wrong** answer a long, fluent, logically inert reasoning chain |
| H6 | `abstention` | **behavioral** | Replace the **correct** answer with a calibrated "I don't know" |
| H7 | `self_preference` | **metadata** | Tag the **wrong** answer with the judge's own model-family source label |

- **H5 is load-bearing**: if extended reasoning makes a *wrong* answer score higher, test-time scaling is degrading evaluation, not just generation. It ships with a length-matched ablation control, because a longer chain-of-thought is necessarily longer — without the control, a positive H5 could be length bias in a reasoning costume.
- **H6 is a behavioral intervention, not a surface perturbation**: it changes what counts as "correct" (calibrated abstention beats confident fabrication). It measures abstention robustness.
- **H7 is attribution / identity-label bias, not true self-preference**: the judge never sees its own outputs — only a `[source: …]` provenance tag. It measures whether a labelled source sways verdicts.

---

## Calibration

A bias detector that has never recovered a known bias is an opinion generator. `scripts/selfcheck.py` runs both directions against simulated judges:

```
$ python3 scripts/selfcheck.py          # ~40s, no network

CONDITION A — judge with NO injected bias (false-positive control)
  [PASS] unbiased judge / position: no false positive     net_bias=-0.006 p_adj=1.000
  ... (all 8 axes silent)
  [PASS] family-wise false positive rate under zero bias  0/24 = 0.0%
         (calibrated-test acceptance region at alpha=0.05: <= 3/24)

CONDITION B — judge WITH injected bias (sensitivity control)
  [PASS] biased judge / position: bias recovered   net_bias=+0.250 h=+0.752 p_adj=0.0007
  [PASS] biased judge / verbose_cot: bias recovered  net_bias=+0.398 h=+1.048 p_adj=0.0007
  [PASS] H5 ablation: reasoning structure beats length alone  delta +0.283

28/28 checks passed
```

The FPR gate deserves a note: with 24 tests at α = 0.05, a *perfectly* calibrated test produces two false positives about a third of the time, so the gate uses the binomial acceptance region (≤ 3/24), not a point estimate. A gate that flaky would test the gate, not the instrument.

---

## Quick start

No dependencies for the core — standard library only, Python 3.10+.

```bash
git clone https://github.com/richardgshen-hub/judge-redteam && cd judge-redteam
python3 -m pip install -e ".[dev]"

python3 scripts/demo.py                                      # end-to-end, no API key, ~6s
python3 scripts/validate_items.py data/items.jsonl --strict  # confound gate on the item pool
python3 scripts/selfcheck.py                                 # instrument calibration
python3 -m pytest tests/ -q                                  # 90 tests
```

`demo.py` runs against a judge with deliberately injected bias: full reporting path (tables, effect sizes, noise floor, disclosure card) at zero cost. Its output describes the simulated judge, not any real model.

### Running real judges

The default backend is **local Ollama** — no paid API is ever contacted by default.

```bash
# cheap pilot first (2 axes, 1 rep) — check the plumbing, not for findings
python3 scripts/run_experiment.py --pilot --judge ollama --model qwen2.5:7b

# formal run — always prints the estimated call count first (e.g. 21,000)
python3 scripts/run_experiment.py --judge ollama --model qwen2.5:7b

# paid backends refuse to start without explicit consent
python3 scripts/run_experiment.py --dry-run --judge openai --model gpt-4o-mini  # plan only
python3 scripts/run_experiment.py --judge openai --model gpt-4o-mini --yes      # explicit consent
```

Every judgment is persisted (and fsynced) as it completes; re-running the same command resumes instead of re-billing. Each run emits `<run>_raw.jsonl`, `<run>_manifest.json` (config + data hash + commit), `<run>_summary.json`, and `<run>_report.md` with full harness disclosure. Transient API failures (429/5xx/timeout) are retried with capped exponential backoff; API keys are redacted from stored error strings.

---

## Honest limitations

1. **No real-model results exist yet.** Everything here is instrument calibration on simulated judges.
2. **150 items covers moderate effects under a conservative cluster model.** The v0.1 power numbers treated replicates as independent and were retired when inference moved to the item level. `scripts/power_analysis.py` now treats perfectly correlated replicates as one item-level unit; its seeded simulation estimates an 80%-power threshold around `h = 0.46`. A null always means "not detected at this resolution", never "absent".
3. **Items are hand-written, not independently validated**, and not checked for training-data contamination in the judges being tested.
4. **Length balance is enforced, but only on length** — not fluency, hedging, or vocabulary sophistication. Strongest known residual confound.
5. **The leakage guard is heuristic** (`audit_leakage` catches new numbers/content words, not subtle semantic strengthening). The protocol commits to a 10% human audit sample — not yet done.
6. **The cluster-permutation statistics have not been reviewed by a statistician** (`stats.STATS_REVIEW_NOTE`). Treat future real-model findings as provisional until then.
7. **Pairwise comparison only**; pointwise scoring is a different regime.
8. **Commercial judges are non-stationary**; every run records model version strings and timestamps for this reason.

## Roadmap

- [ ] Run the preregistered experiment against ≥ 2 real judges (local + hosted), publish raw + report whatever the result
- [ ] 10% human audit of the item pool / perturbation outputs (committed in the protocol, not done)
- [ ] External review of the statistical method (cluster permutation + noise-floor gate)
- [ ] External preregistration (OSF or similar) before the real-model run
- [ ] Item pool growth toward small-effect sensitivity (h ≈ 0.15, ~250 items)

## Related work

Real, clickable, and worth reading — both document judge biases that motivate the direction-decomposed design:

- Zheng, Lian et al. (2023). [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). NeurIPS 2023 Datasets & Benchmarks. Documents position bias and verbosity bias in LLM judges.
- Wang, Peiyi et al. (2023). [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926). Documents how the order of presented answers skews LLM evaluation.

This repo differs from both in that it is a preregistered *audit protocol* with a noise-floor control and item-cluster inference, rather than an observation of bias in a particular system.

## Layout

```
PREREGISTRATION.md   hypotheses, protocol, statistics — locked before data collection
DEVIATIONS.md        dated log of departures from the above (8 amendments so far)
src/jrt/
  types.py           data structures; ground truth is per-presentation, not per-candidate
  axes.py            the seven perturbation axes + H5 ablation control + taxonomy
  stats.py           item-cluster permutation, noise-floor logic, Cohen's h,
                     cluster bootstrap, Holm, DerSimonian-Laird meta-analysis
  judges/            ollama, OpenAI-compatible, Anthropic (retry/backoff, key
                     redaction) + simulated judge with injectable bias
  runner.py          orchestration, immutable run ids, manifest, crash-safe resume
  report.py          results tables, per-template breakdown, disclosure card
data/raw/<domain>.jsonl  item sources, one file per domain; independent audit pending
data/items.jsonl         the assembled 150-item pool (do not hand-edit)
scripts/             build_pool / validate_items / selfcheck / power_analysis /
                     run_experiment / generate_items / demo
```

## Disclosure commitments

- All seven hypotheses reported regardless of outcome. No silent dropping, no post-hoc axes, no subgroup mining.
- Negative and null results published with equal prominence.
- Excluded verdicts (parse failures, ties, transport errors) are counted in every report; raw responses are always preserved.
- Departures from the preregistration logged in `DEVIATIONS.md`, each stating whether it was made before or after seeing outcome data.
- Complete harness specification, manifest (config + item-set hash + code commit), and raw responses ship with any write-up.

## AI-assisted development disclosure

This project was developed with substantial AI assistance (code, tests, item drafting, and documentation), directed and reviewed by the human author. Safeguards against the obvious failure mode — an AI "confirming" its own work — include the preregistration, the dated deviations log, simulated-judge calibration gates that must pass before any real run, and a planned 10% human audit. Method-level statistical decisions are flagged for human expert review (see `stats.STATS_REVIEW_NOTE`).

---

MIT license — see [LICENSE](LICENSE). Hypotheses preregistered 2026-08-30.
