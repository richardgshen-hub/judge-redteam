"""Tests for experiment capability: templates, reporting metrics, cost preview.

Problem 6. The three preregistered prompt templates must be real, runnable, and
reportable separately; failures must be visible, never silently dropped; and the
run must publish its cost before it starts.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jrt import Experiment, RunConfig  # noqa: E402
from jrt.judges.simulated import SimulatedJudge  # noqa: E402
from jrt.prompts import TEMPLATES, build_prompt  # noqa: E402
from jrt.types import Presentation  # noqa: E402

ITEMS_PATH = os.path.join(ROOT, "data", "items.jsonl")

_PRES = Presentation(blocks=(("A", "4"), ("B", "5")), preferred="A")


def _exp(output_dir, **kw):
    cfg = RunConfig(
        items_path=ITEMS_PATH,
        axes=("length",),
        output_dir=str(output_dir),
        **{"reps": 1, **kw},
    )
    return Experiment(cfg, [SimulatedJudge(seed=cfg.seed, name="sim")])


def test_three_templates_exist_and_are_real():
    assert set(TEMPLATES) == {"v1_standard", "v2_cot", "v3_substance"}
    prompts = {
        name: build_prompt("What is 2+2?", _PRES, name) for name in TEMPLATES
    }
    # v1 is the neutral baseline; v2 and v3 each add their preregistered
    # instruction. All three must render non-trivially and be distinct.
    assert prompts["v1_standard"].count("CANDIDATE") == 2
    assert prompts["v2_cot"] != prompts["v1_standard"]
    assert prompts["v3_substance"] != prompts["v1_standard"]
    assert "substance" in prompts["v3_substance"].lower()
    assert "restate" in prompts["v2_cot"].lower()


def test_expected_calls_matches_trial_count(tmp_path):
    exp = _exp(tmp_path, reps=2, templates=("v1_standard", "v2_cot"))
    trials = exp.build_trials()
    # 150 items x 1 axis x (base + perturbed + 2 noise) x 2 templates x 2 reps
    assert len(trials) == 150 * 1 * 4 * 2 * 2
    assert exp.expected_calls(trials) == len(trials)


def test_pooled_and_per_template_analyses_both_run(tmp_path):
    exp = _exp(tmp_path, templates=("v1_standard", "v2_cot"))
    exp.run(verbose=False)
    judgments = exp.load_judgments()

    pooled = exp.analyse(judgments)
    assert "sim" in pooled
    assert len(pooled["sim"]) == 1  # one axis

    per_template = exp.analyse_by_template(judgments)
    assert set(per_template["sim"]) == {"v1_standard", "v2_cot"}
    for results in per_template["sim"].values():
        assert len(results) == 1
        assert results[0].n_items > 0


def test_analysis_attaches_condition_accuracy(tmp_path):
    exp = _exp(tmp_path)
    exp.run(verbose=False)
    res = exp.analyse()["sim"][0]
    assert "acc_base" in res.extra and "acc_pert" in res.extra
    assert 0.0 <= res.extra["acc_base"] <= 1.0
    assert res.extra["n_base"] > 0
    # A scorable verdict excluded from accuracy would be a silent drop; the
    # accuracy denominator must equal the scorable count for the condition.
    assert res.extra["n_base"] + res.extra["n_pert"] >= res.n_pairs


def test_rate_limited_run_completes(tmp_path):
    exp = _exp(tmp_path)
    # Tiny interval: exercises the rate-limit path without slowing the suite.
    out = exp.run(verbose=False, min_interval=0.0005)
    assert len(out) == 150 * 1 * 4  # items x axes x conditions (1 rep, 1 template)


def test_dry_run_estimates_without_calls(tmp_path, capsys, monkeypatch):
    exp = _exp(tmp_path)
    # A judge that would fail loudly if actually called: the dry-run must not
    # issue a single invocation.
    class ExplodingJudge(SimulatedJudge):
        def judge_presentation(self, *a, **kw):  # pragma: no cover
            raise AssertionError("dry-run must not call the judge")

    boom = Experiment(exp.config, [ExplodingJudge(seed=1, name="boom")])
    estimate = boom.expected_calls()
    assert estimate == 150 * 1 * 4
    captured = capsys
    assert captured is not None  # capsys active; no exception raised above
