"""Keep legacy statistic semantics and missing outcomes out of published figures."""

import copy
import json

import pytest

from jrt.figures import public_source


@pytest.fixture
def current_summary():
    return {
        "run_id": "provenance-check",
        "generated_utc": "2026-09-05T00:00:00+00:00",
        "n_items": 50,
        "config": {
            "axes": ["position"],
            "reps": 2,
            "seed": 7,
            "include_noise_floor": True,
        },
        "manifest": {
            "schema": 3,
            "items_sha256": "a" * 64,
            "code_commit": "123456789abc",
            "judges": [{"id": "sim", "type": "SimulatedJudge"}],
        },
        "results": {
            "sim": [{
                "axis": "position",
                "n_pairs": 100,
                "n_items": 50,
                "b": 20,
                "c": 10,
                "net_bias": 0.1,
                "cohens_h": 0.28379410920832787,
                "h_ci": [0.1, 0.5],
                "p_flip_wrong": 0.2,
                "p_flip_right": 0.1,
                "noise_floor": 0.07,
                "above_noise": True,
                "confirmatory": True,
            }],
        },
    }


@pytest.mark.parametrize("schema", [None, 0, 1, 2, 4, "unknown"])
def test_full_summary_rejects_old_or_unknown_statistical_schema(current_summary, schema):
    # In schema 2, h_ci described net bias rather than Cohen's h. Accepting
    # similarly shaped rows would silently put the interval on the wrong scale.
    current_summary["manifest"]["schema"] = schema
    with pytest.raises(ValueError):
        public_source(current_summary)


def test_full_summary_requires_an_explicit_statistical_schema(current_summary):
    del current_summary["manifest"]["schema"]
    with pytest.raises(ValueError):
        public_source(current_summary)


def test_public_export_retains_supported_manifest_schema(current_summary):
    source = public_source(current_summary)
    assert source["schema"] == "jrt-figure-source-v1"
    assert source["manifest"]["schema"] == 3


def test_configured_axis_without_scorable_pairs_is_retained(current_summary):
    current_summary["config"]["axes"].append("length")
    source = public_source(current_summary)
    rows = source["results"]["sim"]
    assert [row["axis"] for row in rows] == ["position", "length"]
    missing = rows[1]
    assert missing["n_pairs"] == 0
    assert missing["n_items"] == 0
    assert missing["cohens_h"] is None
    assert missing["h_ci"] == [None, None]
    assert missing["p_flip_wrong"] is None
    assert missing["p_flip_right"] is None
    assert missing["above_noise"] is None
    # An observed axis is retained rather than overwritten by the placeholder.
    assert rows[0]["n_pairs"] == 100
    assert rows[0]["h_ci"] == [0.1, 0.5]


def test_public_snapshot_roundtrip_preserves_all_exported_data(current_summary):
    current_summary["config"]["axes"].append("length")
    original = copy.deepcopy(current_summary)
    source = public_source(current_summary)
    decoded = json.loads(json.dumps(source, allow_nan=False))
    assert public_source(decoded) == source
    assert current_summary == original


def test_existing_v1_public_snapshot_without_manifest_schema_is_accepted(current_summary):
    source = public_source(current_summary)
    del source["manifest"]["schema"]
    # The first public snapshots used this explicit figure schema, but did not
    # yet copy the full run's manifest schema into their allowlisted metadata.
    restored = public_source(source)
    assert restored["manifest"]["schema"] == 3
    assert restored["results"] == source["results"]
    assert public_source(restored) == restored
