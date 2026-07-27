import json

import pytest

from flagquantum.benchmarking.contract import (
    runtime_metadata,
    validate_payload,
    write_json_atomic,
)


def test_runtime_metadata_is_json_safe(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    payload = runtime_metadata(runner="smoke", schema="test.v1", seed=7)
    assert payload["world_size"] == 4
    json.dumps(payload)


def test_write_json_atomic_replaces_target(tmp_path):
    target = tmp_path / "nested" / "result.json"
    assert (
        write_json_atomic(target, {"runner": "x", "schema": "x.v1", "a": 1}) == target
    )
    assert json.loads(target.read_text()) == {"runner": "x", "schema": "x.v1", "a": 1}
    assert not list(target.parent.glob(".*.result.json.*"))


def test_validate_payload_requires_identity_fields():
    validate_payload({"runner": "x", "schema": "x.v1"})
    with pytest.raises(ValueError, match="runner"):
        validate_payload({"schema": "x.v1"})
    with pytest.raises(ValueError, match="end with .vN"):
        validate_payload({"runner": "x", "schema": "unversioned"})
