from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

pytestmark = pytest.mark.unit


def test_strict_json_accepts_unambiguous_nested_data() -> None:
    assert loads_json_strict('{"outer": {"value": 1}, "items": [true, null]}') == {
        "outer": {"value": 1},
        "items": [True, None],
    }


@pytest.mark.parametrize(
    "encoded",
    (
        '{"schema": "first", "schema": "second"}',
        '{"component": {"sha256": "first", "sha256": "second"}}',
        '{"items": [{"host": "jp-a800-171", "host": "jp-a800-172"}]}',
    ),
)
def test_strict_json_rejects_duplicate_keys_at_every_depth(encoded: str) -> None:
    with pytest.raises(ValueError, match="duplicate object keys"):
        loads_json_strict(encoded)


def test_strict_json_preserves_syntax_failure_type() -> None:
    with pytest.raises(json.JSONDecodeError):
        loads_json_strict('{"missing":')


@pytest.mark.parametrize(
    "encoded",
    (
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e999}',
        '{"nested": [{"value": -1e999}]}',
    ),
)
def test_strict_json_rejects_nonfinite_numbers_at_every_depth(encoded: str) -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        loads_json_strict(encoded)


def test_qdiffusion_evidence_readers_do_not_use_ambiguous_json_loads() -> None:
    source_root = Path(__file__).parents[3] / "examples" / "qdiffusion_kaiwu"
    violations = []
    for path in sorted(source_root.glob("*.py")):
        if path.name == "strict_json.py":
            continue
        if "json.loads(" in path.read_text(encoding="utf-8"):
            violations.append(path.name)

    assert violations == []
