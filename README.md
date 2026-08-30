# judge-redteam

**Auditing LLM-as-Judge under surface-form perturbation.**

Everyone grades models with models now. RLHF pipelines, agent benchmarks,
leaderboards, data curation — all of it routes through an LLM judge. This repo
attacks the judge: it perturbs the *presentation* of an answer without touching
its *substance*, and measures whether the judge can be steered away from ground
truth by nothing but formatting, length, tone, or a confident-looking chain of
reasoning that says nothing.

---

## The one idea that matters

Prior work on judge bias reports **disagreement rates**. That is the wrong
number. A judge with high sampling variance disagrees with itself constantly
under no perturbation at all — and that disagreement is symmetric, destroying
as many correct verdicts as it accidentally repairs.

So every flip here is decomposed by direction:

```
P_flip→wrong = P(base correct     ∧  perturbed wrong)
P_flip→right = P(base wrong       ∧  perturbed correct)

net_bias     = P_flip→wrong − P_flip→right
```

Under pure noise the flips are direction-symmetric and `net_bias = 0`. A
significantly positive `net_bias` means the perturbation is not merely
destabilising the judge — **it is steering it**. That is the difference between
"this judge is noisy" and "this judge is biased", and it is computable without
any assumption about the judge's internals.

A second guard: an axis only counts as a finding if it flips more verdicts than
the judge flips on its own between two unperturbed repeats. Anything below that
self-consistency floor is reported as *not detected*, not as a discovery.

---

## Why this, why now

- **BenchJack** (UC Berkeley, 2026) showed ten major agent benchmarks could be
  driven to near-perfect scores by attacking the scoring *harness*. SWE-bench
  Verified fell to a ten-line `conftest.py`.
- **OpenAI stopped reporting SWE-bench Verified**, citing flawed test cases in a
  majority of the hardest unsolved problems.
- **"Show Your Work"** is now an expected disclosure standard: publish the
  harness, the scaffold, and the negative results, or the number is marketing.

Those works attack the pipeline. This one attacks the scorer. A benchmark can
have an impeccable harness and still emit meaningless numbers if the judge
grading it is biased.

---

## Hypotheses

Seven preregistered, direction-decomposed, Holm-corrected. Full protocol in
[PREREGISTRATION.md](PREREGISTRATION.md).

| ID | Axis | Perturbation |
|----|------|--------------|
| H1 | `position` | Swap the A/B presentation order |
| H2 | `length` | Pad the **wrong** answer with content-free elaboration |
| H3 | `authority` | Prepend assertive rhetoric to the **wrong** answer |
| H4 | `format` | Render the **wrong** answer as structured Markdown |
| H5 | `verbose_cot` | Give the **wrong** answer a long, fluent, logically inert reasoning chain |
| H6 | `abstention` | Replace the **correct** answer with a calibrated "I don't know" |
| H7 | `self_preference` | Attribute the **wrong** answer to the judge's own model family |

**H5 is load-bearing.** If extended reasoning makes a *wrong* answer score
higher, then test-time scaling is degrading evaluation, not just generation.

**H6 is the one with teeth.** If judges rate calibrated abstention *below*
confident fabrication, then every judge-driven RLHF pipeline is actively
optimising against honest uncertainty.

H5 ships with an ablation control (`length_matched_control`) that pads the wrong
answer to the same length with text that says nothing, because a longer chain of
thought is necessarily longer — without the control, a positive H5 could just be
length bias in a reasoning costume.

---

## The instrument is calibrated

A bias detector that has never been shown to recover a known bias is not a
measurement instrument, it is an opinion generator. `scripts/selfcheck.py` runs
both directions against a simulated judge with injected bias:

```
$ python scripts/selfcheck.py

CONDITION A — judge with NO injected bias (false-positive control)
  [PASS] family-wise false positive rate under zero bias   0/24 = 0.0% (target <= 5%)

CONDITION B — judge WITH injected bias (sensitivity control)
  [PASS] biased judge / position: bias recovered           net_bias=+0.254 p_adj=0.0000
  [PASS] biased judge / length: bias recovered             net_bias=+0.151 p_adj=0.0005
  [PASS] biased judge / authority: bias recovered          net_bias=+0.145 p_adj=0.0009
  [PASS] biased judge / format: bias recovered             net_bias=+0.106 p_adj=0.0066
  [PASS] biased judge / verbose_cot: bias recovered        net_bias=+0.343 p_adj=0.0000
  [PASS] biased judge / length_matched_control             net_bias=+0.165 p_adj=0.0000
  [PASS] biased judge / abstention: bias recovered         net_bias=+0.125 p_adj=0.0055
  [PASS] biased judge / self_preference: bias recovered    net_bias=+0.129 p_adj=0.0055
  [PASS] H5 ablation: reasoning structure beats length alone  delta +0.178

28/28 checks passed
Instrument calibrated: recovers known bias, silent when bias is absent.
```

Two things worth reading closely in that output. The false-positive rate is
estimated over 24 tests across three seeds, because one run cannot distinguish a
calibrated instrument from a lucky one — and on individual runs an uncorrected
p-value near 0.05 *does* show up and *is* correctly caught by Holm. Second, the
H5 ablation reports a delta of +0.178 attributable to reasoning structure rather
than length, which is the number H5 actually rests on.

---

## Install and run

No dependencies for the core. Standard library only.

```bash
git clone <your-repo> && cd judge-redteam
python3 -m pip install -e ".[dev]"

python3 scripts/selfcheck.py        # calibrate the instrument
python3 scripts/power_analysis.py   # how many items do you actually need
python3 -m pytest tests/ -q         # 34 tests
```

Against real judges:

```bash
# local, free
ollama pull qwen2.5:7b
python3 scripts/run_experiment.py --judge ollama --model qwen2.5:7b

# hosted
export OPENAI_API_KEY=...
python3 scripts/run_experiment.py --judge openai --model gpt-4o-mini

export ANTHROPIC_API_KEY=...
python3 scripts/run_experiment.py --judge anthropic --model claude-opus-4-6 --reps 5
```

Every judgment is written to disk as it completes. Re-running the same command
resumes instead of restarting, which matters when the run costs money. Each run
emits `<run>_raw.jsonl`, `<run>_summary.json`, and `<run>_report.md`, where the
report carries the full harness disclosure — temperature, prompt templates,
replicate count, model version strings, parse-failure rate.

---

## Honest limitations

Read these before quoting any number this repo produces.

1. **The item pool is 150 items, which covers medium effects and not small
   ones.** Measured power at a 15% noise floor: 150 items detect `h ≥ ~0.195` at
   80% power; detecting `h = 0.15` would need roughly 250. **A null result means
   "not detected at this resolution", never "absent"** — and any effect below
   `h ≈ 0.15` is simply invisible to this design.
2. **The items are hand-written and not independently validated.** They have not
   been checked for contamination in the training data of the judges being
   tested. Arithmetic and factual items are the most exposed to this.
3. **Length balance is enforced, but only on length.** The pool is matched on
   character count, digit density, and line count (mean signed gap +0.003). It is
   *not* matched on fluency, hedging, or vocabulary sophistication, any of which
   could leak a cue in the same way. This is the strongest known residual
   confound.
4. **Pairwise comparison only.** Pointwise scoring, which many pipelines use, is
   a different regime and may behave differently.
5. **The perturbation guard is heuristic.** `audit_leakage` catches new numbers
   and new content words, not subtle semantic strengthening. A 10% human audit
   sample is committed in the protocol and is not yet done.
6. **Commercial judges are non-stationary.** Results are a snapshot. Every run
   records model version strings and a timestamp for this reason.

---

## Growing the item pool

Sources live one file per domain in `data/raw/`, so a domain can be extended
without touching the others. After editing a source, rebuild and re-gate:

```bash
python3 scripts/build_pool.py                              # -> data/items.jsonl
python3 scripts/validate_items.py data/items.jsonl --strict # hard gate
python3 -m pytest tests/ -q                                 # same rules as invariants
```

Write items to a **symmetric template**: identical reasoning skeleton, one step
divergent. That makes length balance a property of construction rather than
something you trim into afterwards, and it is the only reason the current pool
clears the gate at +0.003 mean gap.

An AI generator is provided, but treat it as a source of drafts:

```bash
python3 scripts/generate_items.py generate --domain arithmetic --n 40 --out data/candidates.jsonl
python3 scripts/generate_items.py audit --inp data/candidates.jsonl --out data/accepted.jsonl
```

The generator proposes, the filters dispose, and **a human still has to read
every item before it enters the pool**. Letting a model write the eval set
unsupervised is precisely how benchmarks became untrustworthy in the first place.

---

## Layout

```
PREREGISTRATION.md   hypotheses, protocol, statistics — locked before data collection
DEVIATIONS.md        dated log of any departure from the above
src/jrt/
  types.py           data structures; ground truth is per-presentation, not per-candidate
  axes.py            the seven perturbation axes plus the H5 ablation control
  stats.py           exact McNemar, net-bias decomposition, Cohen's h,
                     cluster bootstrap, Holm, DerSimonian-Laird meta-analysis
  judges/            ollama, OpenAI-compatible, Anthropic, and a simulated
                     judge with injectable bias for calibration
  runner.py          orchestration, raw persistence, crash-safe resume
  report.py          results tables and Show-Your-Work disclosure card
data/raw/<domain>.jsonl  item sources, one file per domain, human-audited
data/items.jsonl         the assembled 150-item pool (do not hand-edit)
scripts/
  build_pool.py      merge sources into the pool, interleaving domains
  validate_items.py  confound gate: length balance, leakage cues, schema
  selfcheck.py       instrument calibration against known injected bias
  power_analysis.py  Monte Carlo power over the actual test statistic
  run_experiment.py  run against real judges (costs money, resumes)
  generate_items.py  AI-assisted drafting of new candidates
  demo.py            end-to-end run on the simulated judge, no API key
```

---

## Disclosure commitments

- All seven hypotheses reported regardless of outcome. No silent dropping, no
  post-hoc axes, no subgroup mining.
- Negative and null results published with equal prominence.
- Complete harness specification and raw responses shipped with any write-up.
- Departures from the preregistration logged in `DEVIATIONS.md`, each stating
  whether it was made before or after seeing outcome data.

---

MIT license. Hypotheses preregistered 2026-08-30.
