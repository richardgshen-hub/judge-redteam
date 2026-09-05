"""Protect figure provenance, denominators and report regeneration boundaries."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from jrt.figures import START, embed_figures, public_source, render_figures

ITEMS = Path(__file__).resolve().parents[1] / "data/items.jsonl"


@pytest.fixture
def summary():
    return {
        "run_id": "fixture", "n_items": 150,
        "config": {"seed": 7, "include_noise_floor": True, "items_path": "/private/author/data", "api_key": "secret"},
        "manifest": {"items_sha256": hashlib.sha256(ITEMS.read_bytes()).hexdigest(), "code_commit": "abc", "judges": [{"id": "sim", "type": "SimulatedJudge", "api_key": "secret"}]},
        "results": {"sim": [
            {"axis": "position", "n_pairs": 100, "n_items": 50, "b": 20, "c": 10, "cohens_h": .3, "h_ci": [.1, .5], "p_flip_wrong": .2, "p_flip_right": .1, "noise_floor": .07, "confirmatory": True},
            {"axis": "length_matched_control", "n_pairs": 80, "n_items": 40, "cohens_h": -.1, "h_ci": [-.3, .1], "p_flip_wrong": .05, "p_flip_right": .1, "noise_floor": .06, "confirmatory": False},
        ]},
    }


def test_public_source_strips_private_fields_and_encodes_missing_intervals(summary):
    summary["results"]["sim"][0]["h_ci"] = [float("nan"), float("nan")]
    source = public_source(summary)
    encoded = json.dumps(source, allow_nan=False)
    assert "/private" not in encoded and "secret" not in encoded
    assert source["results"]["sim"][0]["h_ci"] == [None, None]
    assert source["results"]["sim"][1]["confirmatory"] is False


def test_multiple_judges_require_explicit_selection(summary):
    summary["results"]["second"] = copy.deepcopy(summary["results"]["sim"])
    with pytest.raises(ValueError, match="Multiple judges"):
        public_source(summary)
    assert list(public_source(summary, "second")["results"]) == ["second"]


def test_report_embedding_is_idempotent_and_preserves_user_text(tmp_path):
    tmp_path = Path(tmp_path)
    report = tmp_path / "report.md"
    report.write_text("# Report\n\nMy explanation.\n\n## Harness disclosure\n\nExisting table.\n", encoding="utf-8")
    embed_figures(report, tmp_path / "figures")
    once = report.read_text()
    embed_figures(report, tmp_path / "figures")
    assert report.read_text() == once
    assert once.count(START) == 1
    assert "My explanation." in once and "Existing table." in once
    assert 'src="figures/effect_sizes.svg"' in once


def test_renderer_refuses_mismatched_item_pool(summary, tmp_path):
    tmp_path = Path(tmp_path)
    pytest.importorskip("matplotlib")
    summary["manifest"]["items_sha256"] = "not-this-pool"
    path = tmp_path / "source.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        render_figures(path, ITEMS, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_figures_and_csv_preserve_negative_control_and_observed_rates(summary, tmp_path):
    tmp_path = Path(tmp_path)
    pytest.importorskip("matplotlib")
    summary["results"]["sim"][0]["h_ci"] = [None, None]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "figures"
    files = render_figures(path, ITEMS, output)
    assert len(files) == 8
    assert len(list(output.glob("*.png"))) == 4
    assert all(p.stat().st_size > 1000 for p in files)
    with (output / "axis_results.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[0]["p_flip_wrong"]) == .2
    assert float(rows[1]["cohens_h"]) == -.1
    assert rows[0]["h_ci_low"] == ""
    with (output / "item_pool.csv").open(newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 150
