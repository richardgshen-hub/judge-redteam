"""Tests for run identity, persistence, and crash-safe resume.

The data-integrity guarantees: one immutable run id, honest resume that never
mixes two experiments, de-duplication of any double-written rows, and a
machine-readable manifest that records exactly what was run.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jrt import Experiment, RunConfig  # noqa: E402
from jrt.runner import ConfigConflict  # noqa: E402
from jrt.judges.simulated import SimulatedJudge  # noqa: E402

ITEMS_PATH = os.path.join(ROOT, "data", "items.jsonl")

# tmp_path (overridden in conftest.py to a plain /tmp directory) is used by the
# resume/integrity tests below.


def _exp(output_dir, axes=("length",), reps=2, seed=20260830, **kw):
    cfg = RunConfig(
        items_path=ITEMS_PATH,
        axes=axes,
        reps=reps,
        seed=seed,
        output_dir=str(output_dir),
        **kw,
    )
    return Experiment(cfg, [SimulatedJudge(seed=seed, name="sim")])


# ---------------------------------------------------------- run identity


def test_run_id_is_computed_once_and_immutable():
    e = _exp("/tmp/throwaway-x1")
    first = e.name
    # name is a property backed by a single computed id, not wall-clock time.
    again = e.name
    assert first == again
    assert first  # non-empty


def test_run_id_is_stable_across_instances_same_config(tmp_path):
    e1 = _exp(tmp_path, seed=7)
    e2 = _exp(tmp_path, seed=7)
    assert e1.name == e2.name


def test_run_id_changes_when_config_changes(tmp_path):
    e1 = _exp(tmp_path, seed=7)
    e2 = _exp(tmp_path, seed=8)
    assert e1.name != e2.name


def test_axis_order_is_part_of_run_identity(tmp_path):
    e1 = _exp(tmp_path, axes=("length", "authority"))
    e2 = _exp(tmp_path, axes=("authority", "length"))
    assert e1.name != e2.name


def test_trial_construction_is_idempotent(tmp_path):
    e = _exp(tmp_path, axes=("authority", "format"))
    first = e.build_trials()
    second = e.build_trials()
    assert first == second


def test_explicit_run_name_is_preserved(tmp_path):
    e = _exp(tmp_path, run_name="my-run")
    assert e.name == "my-run"


# ---------------------------------------------------------- resume / integrity


def test_resume_restores_without_duplication(tmp_path):
    e1 = _exp(tmp_path, axes=("length",), reps=2)
    total = len(e1.build_trials())
    e1.run(verbose=False)

    e2 = _exp(tmp_path, axes=("length",), reps=2)
    pending = [t for t in e2.build_trials() if t.key not in e2._done_keys()]
    assert pending == [], "a full run should leave nothing pending on resume"
    e2.run(verbose=False)

    all_j = e2.load_judgments()
    assert len(all_j) == total, "resumed run must not duplicate or lose rows"


def test_resume_preserves_original_created_timestamp(tmp_path):
    e1 = _exp(tmp_path)
    e1.run(verbose=False)
    original = e1.load_manifest().created_utc

    e2 = _exp(tmp_path)
    e2.manifest.created_utc = "2099-01-01T00:00:00+00:00"
    e2.run(verbose=False)
    assert e2.load_manifest().created_utc == original


def test_resume_rejects_config_change(tmp_path):
    e1 = _exp(tmp_path, seed=1, run_name="shared")
    e1.run(verbose=False)

    # Same explicit run name but a different seed is a different experiment.
    # Resuming would silently mix two incompatible datasets, so it must refuse.
    e2 = _exp(tmp_path, seed=2, run_name="shared")
    with pytest.raises(ConfigConflict):
        e2.run(verbose=False)


def test_resume_rejects_without_manifest(tmp_path):
    e1 = _exp(tmp_path, seed=1)
    e1.run(verbose=False)
    # Simulate an old-style run that has data but no manifest.
    os.remove(e1.manifest_path)

    e2 = _exp(tmp_path, seed=1)
    with pytest.raises(ConfigConflict):
        e2.run(verbose=False)


def test_load_judgments_dedups_corrupt_double_write(tmp_path):
    e1 = _exp(tmp_path, axes=("length",), reps=2)
    e1.run(verbose=False)
    before = len(e1.load_judgments())

    # Corrupt the file by duplicating its first line, as a crash mid-flush could.
    with open(e1.raw_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    with open(e1.raw_path, "w", encoding="utf-8") as fh:
        fh.write(lines[0])
        fh.writelines(lines)

    e2 = _exp(tmp_path, axes=("length",), reps=2)
    after = len(e2.load_judgments())
    assert after == before, "duplicate rows must be collapsed on load"


def test_truncated_final_line_is_tolerated(tmp_path):
    e1 = _exp(tmp_path, axes=("length",), reps=2)
    e1.run(verbose=False)
    before = len(e1.load_judgments())

    # Truncate the file mid-record, as a kill would.
    with open(e1.raw_path, encoding="utf-8") as fh:
        data = fh.read()
    with open(e1.raw_path, "w", encoding="utf-8") as fh:
        fh.write(data[: len(data) // 2])

    e2 = _exp(tmp_path, axes=("length",), reps=2)
    after = len(e2.load_judgments())
    assert after <= before
    assert after > 0


# ---------------------------------------------------------- manifest


def test_manifest_records_identity(tmp_path):
    e = _exp(tmp_path, axes=("length", "format"), reps=3, seed=42)
    e.run(verbose=False)
    assert os.path.exists(e.manifest_path)
    m = e.load_manifest()
    assert m is not None
    assert m.run_id == e.name
    assert m.items_sha256 == e.items_sha256
    assert m.items_count > 0
    assert m.config["axes"] == ["length", "format"]
    assert m.schema >= 1


def test_manifest_refuses_mixed_when_seed_differs(tmp_path):
    e1 = _exp(tmp_path, seed=1)
    e1.run(verbose=False)
    m1 = e1.load_manifest()

    e2 = _exp(tmp_path, seed=2)
    m2 = e2.manifest
    diffs = m1.diff(m2)
    assert any("seed" in d for d in diffs)


def test_manifest_refuses_different_judge_identity(tmp_path):
    cfg = RunConfig(
        items_path=ITEMS_PATH,
        axes=("length",),
        reps=1,
        output_dir=str(tmp_path),
        run_name="shared",
    )
    e1 = Experiment(cfg, [SimulatedJudge(name="sim", family="Family A")])
    e1.run(verbose=False)
    e2 = Experiment(cfg, [SimulatedJudge(name="sim", family="Family B")])
    with pytest.raises(ConfigConflict):
        e2.run(verbose=False)


def test_manifest_refuses_different_code_commit(tmp_path):
    e1 = _exp(tmp_path, run_name="shared")
    e1.run(verbose=False)

    e2 = _exp(tmp_path, run_name="shared")
    e2.manifest.code_commit = "different-commit"
    with pytest.raises(ConfigConflict, match="code commit"):
        e2.run(verbose=False)


def test_each_completed_call_is_flushed_and_fsynced(tmp_path, monkeypatch):
    e = _exp(tmp_path, axes=("length",), reps=1)
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    out = e.run(verbose=False)
    assert len(calls) >= len(out)
