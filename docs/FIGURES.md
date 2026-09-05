# Report figures

These are static research graphics for GitHub Markdown, not a web application.
The committed demo describes a simulated judge with deliberately injected biases.
It makes no claim about the performance of any real model.

## Reading the figures

| Figure | Question and encoding | Source and interpretation |
|---|---|---|
| [Protocol](figures/demo/protocol.svg) | How does a question become auditable evidence? A five-step flow diagram. | Schematic of `runner.py`, `stats.py` and `report.py`; counts come from this run and its item pool. |
| [Effect sizes](figures/demo/effect_sizes.svg) | How large is each directional effect, and how uncertain? Dots and horizontal intervals share a zero baseline. | `cohens_h` and `h_ci`. Unadjusted 95% item-cluster bootstrap intervals. The grey diamond is the exploratory H5 length control. |
| [Directional flips](figures/demo/directional_flips.svg) | How often does the verdict become wrong or recover? Paired horizontal bars, plus a diamond noise reference. | `p_flip_wrong`, `p_flip_right`, `noise_floor`. Shares of paired scorable trials, not probabilities conditional on an initially correct/wrong answer. |
| [Item pool](figures/demo/item_pool.svg) | Are correct answers systematically longer? One dot per item, grouped by domain, with ±15% reference lines. | Whitespace-normalized character lengths, using the same formula as `validate_items.py`. Vertical displacement only separates marks; some dots may overlap. |

The figure design uses local statistical, diagram and report-export review passes.
Effect intervals and flip rates retain their original analysis units. Blue marks
show effect estimates or item measurements; orange and teal distinguish wrong-
and right-direction flips; neutral diamonds identify controls. Direct labels,
CSV alternatives and portrait SVG variants preserve meaning without color alone.

The displayed aggregate noise dots are descriptive. The decision gate compares
paired **item-level** wrong-flip rates using a bootstrap interval for their
difference. `noise_ci` is that difference interval, not an error bar around
`noise_floor`, and is deliberately not drawn around the noise dots. Condition
accuracies have separate scorable-call denominators and are not substituted for
paired `net_bias`. The exploratory H5 control is outside the seven-test Holm family.

## Regenerate

Install the optional renderer; the core experiment still has no runtime dependencies:

```bash
python -m pip install -e ".[viz]"
python scripts/demo.py --figures
```

This reruns only the simulated demo, updates its Markdown report, and exports the
figures to `docs/figures/demo/`. To recreate the committed figures without running
any experiment, use the small public source snapshot:

```bash
python scripts/render_figures.py \
  --summary docs/figures/demo/source.json \
  --items data/items.jsonl \
  --output docs/figures/demo
```

For your own completed run:

```bash
python scripts/render_figures.py \
  --summary results/my-run_summary.json \
  --items data/items.jsonl \
  --output results/my-run-figures \
  --report results/my-run_report.md
```

Pass `--judge EXACT_ID` if the summary contains multiple judges. Only that judge's
results are rendered. The renderer refuses an item pool whose SHA-256 does not
match the run manifest. `--report` replaces only the marked generated figure
section and preserves the rest of an existing report. Running a report generator
again without `--figures` produces its normal text-only report.

Each export contains four desktop SVGs, four portrait SVGs, four high-resolution
PNGs, `source.json`, `axis_results.csv`, and `item_pool.csv`. Desktop assets are
intended for full-width Markdown; `<picture>` selects the taller variant on narrow
screens. PNGs are available for slides and documents. SVG labels are stored as
outlines for consistent rendering; the surrounding alt text and CSV tables
provide text alternatives.

`source.json` is an allowlisted snapshot of the selected result rows, relevant
run settings, judge ID/type, item-set hash, source timestamp and code commit.
Backend credentials and the manifest's local directory paths are not copied.
Raw responses remain local and are not part of the figure package. Regeneration
uses fixed layout and SVG IDs; use the same Matplotlib version for byte-identical
exports across machines.

## Verification

Figure tests cover source-field filtering, JSON-safe missing intervals, explicit
multi-judge selection, mismatched item hashes, retained negative control effects,
CSV values, export completeness, and repeatable report insertion. CI renders from
the committed public snapshot and uploads the resulting figures as a build
artifact. Visual review checks desktop and portrait labels, clipping, spacing,
zero baselines, method notes and source labels. The statistical method still
requires external expert review; attractive figures do not change that status.
