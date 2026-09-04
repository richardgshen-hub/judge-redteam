# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-08-30

Hardening release: research prototype → structured, reproducible harness.
All changes below were made **before any real-model data collection**; no real
model results exist in this repository.

### Fixed

- **Run identity & resume.** `Experiment.name` no longer recomputes a timestamp
  on every access. Run ids are computed once from a content fingerprint
  (config + item-set hash + judges), a manifest is written before data
  collection, and resuming refuses to mix two different experiments
  (`ConfigConflict`). SIGINT interrupts flush and fsync; truncated final lines
  and double-written rows are tolerated and de-duplicated.
- **Statistics.** The headline test is now an item-cluster permutation test
  (`cluster_permutation_p`) instead of an exact McNemar on pooled discordant
  counts, which treated correlated replicates of one item as independent.
  Simulation tests cover null false-positive rate, power under known bias, and
  perfectly-correlated replicates. Method is flagged as not yet
  statistician-reviewed (`stats.STATS_REVIEW_NOTE`).
- **Noise floor.** `noise_floor` now specifically means the *wrong-direction
  self-flip rate*; the total self-disagreement rate is reported separately
  (`noise_self_disagreement`) and is not used for verdicts. Significant results
  below the noise floor can no longer be reported as findings.
- **Hypothesis naming.** H6 reclassified as a behavioral intervention
  (abstention robustness); H7 reclassified as attribution / identity-label bias
  (metadata) rather than true self-preference. Taxonomy recorded in
  `axes.AXIS_TAXONOMY`; both changes logged as DEVIATIONS.md amendments 5–6.

### Added

- HTTP transport hardening: capped exponential backoff on 429/5xx/timeouts,
  fail-fast on auth errors and malformed bodies, API-key redaction in persisted
  error strings, injectable sleep for deterministic tests.
- CI (GitHub Actions): tests on Python 3.10–3.13, strict data gate, demo smoke
  test.
- MIT `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, this changelog.
- Tests: run identity/resume (11), statistics (10), axes/taxonomy (7), HTTP
  transport (9).

## [0.1.0] — 2026-08-30

- Preregistered protocol, 150-item balanced pool (length-balanced by
  construction after discarding a confounded 40-item pilot — see DEVIATIONS.md
  amendments 1–4).
- Eight perturbation axes (H1–H7 confirmatory + length-matched ablation
  control), three prompt templates, simulated and HTTP judge backends.
- Direction-decomposed reporting (net bias = P(correct→wrong) −
  P(wrong→correct)) with a simulated-noise-floor comparison.
