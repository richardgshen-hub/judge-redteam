# Contributing to judge-redteam

Thanks for wanting to help. This project measures *bias* in LLM judges, which
makes it unusually vulnerable to a specific failure: a well-meaning change that
quietly invalidates comparability. The rules below exist to prevent that.

## Setup

```bash
git clone https://github.com/richardgshen-hub/judge-redteam
cd judge-redteam
python -m pip install -e ".[dev]"
python -m pytest -q                        # all tests must pass
python scripts/validate_items.py data/items.jsonl --strict  # data gate must pass
python scripts/demo.py                     # simulated smoke test, no network
```

The core has **zero runtime dependencies** (standard library only). Do not add
a dependency to the core without a strong justification.

## Ground rules

1. **Do not fabricate results.** Never commit model outputs you did not actually
   collect, and never present simulated-calibration numbers as real-model
   findings. Simulated judges exist to test the harness; reports from them must
   say so.
2. **Method changes go in DEVIATIONS.md.** If your change affects what a
   p-value, an effect size, or a verdict *means*, add a dated amendment stating
   whether it was made before or after any outcome data existed. Same for
   changes to the preregistered hypotheses (H1–H7) or their classification.
3. **The analysis unit is the item.** Replicates of one item are correlated.
   Anything that resamples individual verdicts as if they were independent is a
   bug, not a feature. See `stats.cluster_permutation_p` and
   `tests/test_stats.py`.
4. **Noise floor discipline.** A result below the wrong-direction self-flip rate
   must never be reported as a finding, even when p < alpha. Tests enforce this;
   keep them passing.
5. **Run integrity.** Every run has one immutable identity derived from its
   config and data hash. Resume logic must never silently mix two experiments.
   `tests/test_runner.py` covers this.

## What a good PR looks like

- One logical change per PR, with tests that fail before it and pass after.
- No new runtime dependencies in `src/jrt/`.
- No absolute paths, no secrets, no generated result files committed.
- If it touches statistics, say explicitly in the PR description what you
  verified by simulation and what still needs review.

## Reporting bugs

Open an issue with the command you ran, the config, and the manifest
(`*_manifest.json`) if one exists. Do not paste API keys — including in error
strings; the transport layer redacts them, but check before pasting anyway.
