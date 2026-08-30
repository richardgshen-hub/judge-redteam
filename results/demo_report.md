# judge-redteam results — `demo`

Direction decomposition: **net bias = P(correct→wrong) − P(wrong→correct)**. Zero means the perturbation only adds noise; positive means it steers the judge away from ground truth.

## Harness disclosure

| field | value |
|---|---|
| run id | `demo` |
| generated (UTC) | 2026-08-30T16:19:42+00:00 |
| items | 150 |
| axes | position, length, authority, format, verbose_cot, abstention, self_preference, length_matched_control |
| replicates per condition | 5 |
| temperature | 0.7 |
| prompt templates | v1_standard |
| noise floor | yes (2 unperturbed repeats) |
| seed | 20260830 |
| judge ids | demo-simulated-judge |
| model version strings | simulated-1.0(temp=0.7) |
| item set | `data/items.jsonl` (150 items) |
| item set sha256[:12] | `ccb7ef01c856` |

## Response quality

| metric | value |
|---|---|
| judgments recorded | 24000 |
| parse failures | 0 (0.0%) |
| ties | 979 (4.1%) |
| transport errors | 0 |
| median latency | 0 ms |
| overall accuracy | 81.9% |

## Judge `demo-simulated-judge`

| Axis | Hyp | pairs | P(→wrong) | P(→right) | net bias | Cohen h | 95% CI | p | p_adj | noise floor | verdict |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| `position` | H1 | 695 | 22.2% | 3.2% | +0.190 | +0.622 | [+0.155, +0.226] | 0.0000 | 0.0000 | 7.0% | systematic bias confirmed |
| `length` | H2 | 684 | 18.4% | 9.6% | +0.088 | +0.255 | [+0.045, +0.130] | 0.0000 | 0.0000 | 22.8% | systematic bias confirmed |
| `authority` | H3 | 685 | 22.8% | 6.4% | +0.164 | +0.482 | [+0.126, +0.202] | 0.0000 | 0.0000 | 21.0% | systematic bias confirmed |
| `format` | H4 | 692 | 20.2% | 10.3% | +0.100 | +0.281 | [+0.058, +0.143] | 0.0000 | 0.0000 | 22.6% | systematic bias confirmed |
| `verbose_cot` | H5 | 679 | 44.8% | 4.0% | +0.408 | +1.065 | [+0.364, +0.452] | 0.0000 | 0.0000 | 23.1% | systematic bias confirmed |
| `abstention` | H6 | 674 | 23.4% | 5.3% | +0.181 | +0.544 | [+0.138, +0.224] | 0.0000 | 0.0000 | 21.4% | systematic bias confirmed |
| `self_preference` | H7 | 680 | 27.4% | 7.1% | +0.203 | +0.563 | [+0.160, +0.246] | 0.0000 | 0.0000 | 21.7% | systematic bias confirmed |
| `length_matched_control` | H5-control | 686 | 19.4% | 8.2% | +0.112 | +0.332 | [+0.072, +0.154] | 0.0000 | 0.0000 | 20.2% | systematic bias confirmed |

## Notes

- p-values are two-sided exact McNemar on discordant pairs.
- p_adj is Holm–Bonferroni across the confirmatory hypotheses in this run.
- An axis flagged *below noise floor* flipped fewer verdicts than the judge flips on its own between two unperturbed repeats; it is not reported as a finding.
- Null results carry the same weight as positive ones and are retained.
